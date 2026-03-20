import cv2
import math
from typing import Iterable, Literal, Optional, Union, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class FeatureBank:

    def __init__(
        self,
        train_embeds: Iterable[Union[list, np.ndarray, torch.Tensor]],
        device: Union[str, torch.device] = "cpu",
        eps: float = 1e-8,
    ):

        self.device = torch.device(device)

        N_embeds = len(train_embeds)
        self.bank = None

        # Consume train_embeds and insert directly into preallocated tensor
        # to avoid holding duplicate 10GB+ arrays in RAM during torch.stack
        for idx in range(N_embeds):
            e = train_embeds.pop(0)
            t = self._to_tensor(e)
            # t.shape=[C, H, W]
            if t.dim() == 4 and t.size(0) == 1:
                t = t.squeeze(0)
            t = t.to(self.device, dtype=torch.float16)

            if self.bank is None:
                c, h, w = t.shape
                self.bank = torch.empty(
                    (N_embeds, c, h, w), device=self.device, dtype=torch.float16
                )

            self.bank[idx] = t

        self.N, self.C, self.H, self.W = self.bank.shape
        self._assert_chw(self.bank, self.C, self.H, self.W)

        # Removed unused Mahalanobis distance statistics (mu, var, inv_std) as they
        # duplicate the entire N-sized bank in float32, which directly causes OOM for high-res images.
        # bank_fp32 = self.bank.to(torch.float32)
        # self.mu = bank_fp32.mean(dim=0).to(torch.float16)
        # self.var = (bank_fp32.var(dim=0, unbiased=False) + eps).to(torch.float16)
        # self.inv_std = self.var.to(torch.float32).sqrt().reciprocal().to(torch.float16)
        # self.inv_var = self.var.to(torch.float32).reciprocal().to(torch.float16)

    @staticmethod
    def _to_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    @staticmethod
    def _assert_chw(tensors: Iterable[torch.Tensor], c: int, h: int, w: int):
        for i, t in enumerate(tensors):
            assert t.shape == (
                c,
                h,
                w,
            ), f"Mismatch t[{i}]shape={t.shape} vs. {(c, h, w)}"

    @torch.no_grad()
    def cal_anomaly_map(
        self,
        test_embed: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:

        # test_embed t.shape=[C, H, W]
        t = self._to_tensor(test_embed)
        if t.dim() == 4 and t.size(0) == 1:
            t = t.squeeze(0)

        # Assertion: t.shape==[self.C, self.H, self.W]
        self._assert_chw([t], self.C, self.H, self.W)

        # t.shape=[1, C, H, W]
        t = t.to(self.device, dtype=torch.float32).unsqueeze(0)  # Keep float32 for math

        # b.shape=[n, C, H, W]
        b = self.bank

        t_norm = F.normalize(t, p=2, dim=1)  # [1, C, H, W],

        # Calculate similarities and distances in a loop to drastically reduce peak RAM
        dist_list = []
        for i in range(b.size(0)):
            b_i = b[i : i + 1].to(dtype=torch.float32)  # [1, C, H, W]
            b_norm_i = F.normalize(b_i, p=2, dim=1)
            sim_i = (t_norm * b_norm_i).sum(dim=1, keepdim=True)  # [1, 1, H, W]

            dist_i = 2 * (1.0 - sim_i)
            dist_i = dist_i.clamp_min(
                0.0
            ).sqrt()  # [1, 1, H, W], Euclidean from cosine sim

            # Smooth immediately inside the loop to avoid huge intermediary matrices
            kernel_size = 3
            padding = kernel_size // 2
            dist_i = F.avg_pool2d(
                dist_i, kernel_size=kernel_size, stride=1, padding=padding
            )

            dist_list.append(dist_i.squeeze(0).squeeze(0))  # append as [H, W]
            del b_i, b_norm_i, sim_i  # Ensure we clean up early inside loop

        dists = torch.stack(dist_list, dim=0)  # [n, H, W]
        anomaly, _ = dists.min(dim=0)  # [H, W]

        anomaly = anomaly.detach().cpu().numpy()
        # print(f'max val: {anomaly.max()}, min val: {anomaly.min()}')
        return anomaly


# =============================================================================================================
# Utils for stitching SAM2 features from overlapping patches
def _compute_padded_hw(H, W, patch_size, overlap):
    """
    Pad full image once so that sliding windows (stride=patch-overlap) cover the whole image
    and last window lands exactly at the end (Hp-patch_size is divisible by stride).
    """
    stride = patch_size - overlap
    assert stride > 0, "overlap must be < patch_size"

    def _pad_len(L):
        if L <= patch_size:
            return patch_size
        # need Hp such that: starts = 0, stride, ..., Hp-patch_size covers >= L
        n_steps = math.ceil(
            (L - patch_size) / stride
        )  # how many stride moves after first
        return patch_size + n_steps * stride

    Hp = _pad_len(H)
    Wp = _pad_len(W)
    return Hp, Wp, stride


def _pad_image(full_image, Hp, Wp, pad_value=0):
    H, W, C = full_image.shape
    pad_bottom = Hp - H
    pad_right = Wp - W
    if pad_bottom == 0 and pad_right == 0:
        return full_image
    return np.pad(
        full_image,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )


def _sliding_starts(Lp, patch_size, stride):
    """All top-left starts including the last one at Lp - patch_size."""
    if Lp == patch_size:
        return [0]
    starts = list(range(0, Lp - patch_size + 1, stride))
    if starts[-1] != Lp - patch_size:
        starts.append(Lp - patch_size)
    return starts


def _stitch_feature_level_overlap(
    feat_tiles,
    starts_y,
    starts_x,
    H,
    W,
    patch_size,
    overlap,
    downscale,
):
    """
    Sum + count stitching with overlap averaging.
    feat_tiles: list of Tensor [1, C, ht, wt] in row-major over (starts_y, starts_x)
    returns: Tensor [C, ceil(H/downscale), ceil(W/downscale)]
    """
    assert len(feat_tiles) == len(starts_y) * len(starts_x)

    t0 = feat_tiles[0]
    assert t0.ndim == 4 and t0.shape[0] == 1, f"expect [1,C,h,w], got {tuple(t0.shape)}"
    _, C, ht, wt = t0.shape

    # stitched canvas size on padded image (feature space)
    Hp, Wp, stride = _compute_padded_hw(H, W, patch_size, overlap)
    Hpf, Wpf = math.ceil(Hp / downscale), math.ceil(Wp / downscale)

    device, dtype = t0.device, t0.dtype
    acc = torch.zeros((C, Hpf, Wpf), device=device, dtype=dtype)
    cnt = torch.zeros((1, Hpf, Wpf), device=device, dtype=dtype)

    # map image-space starts to feature-space starts
    # (assumes SAM2 feats align to downscale grid; // is usually correct here)
    idx = 0
    for y in starts_y:
        yf = y // downscale
        for x in starts_x:
            xf = x // downscale
            tile = feat_tiles[idx][0]  # [C, ht, wt]
            acc[:, yf : yf + ht, xf : xf + wt] += tile
            cnt[:, yf : yf + ht, xf : xf + wt] += 1.0
            idx += 1

    # crop back to original (non-padded) size in feature space
    Hout = math.ceil(H / downscale)
    Wout = math.ceil(W / downscale)
    acc = acc[:, :Hout, :Wout]
    cnt = cnt[:, :Hout, :Wout].clamp_min(1.0)

    return acc / cnt


def stitch_sam2_patch_feats(
    predictor,
    full_image: np.ndarray,
    patch_size: int = 1024,
    overlap: int = 64,
    pad_value: int = 0,
):
    """
    - full_image: np.ndarray (H,W,3) RGB
    - overlap: pixels in image space, will be averaged by count in stitched feature maps
    return: Tensor [C_total, ceil(H/4), ceil(W/4)]
    """
    H, W, _ = full_image.shape
    Hp, Wp, stride = _compute_padded_hw(H, W, patch_size, overlap)
    padded = _pad_image(full_image, Hp, Wp, pad_value=pad_value)

    starts_y = _sliding_starts(Hp, patch_size, stride)
    starts_x = _sliding_starts(Wp, patch_size, stride)

    f0_tiles, f1_tiles, f2_tiles = [], [], []
    for y in starts_y:
        for x in starts_x:
            patch = padded[y : y + patch_size, x : x + patch_size, :]
            predictor.set_image(patch)
            f0, f1, f2 = predictor.get_feat_list()  # [1,C,h,w]
            # Immediately detach to CPU to ensure VRAM is perfectly clean
            f0_tiles.append(f0.detach().cpu().half())
            f1_tiles.append(f1.detach().cpu().half())
            f2_tiles.append(f2.detach().cpu().half())

    torch.cuda.empty_cache()

    f0_full = _stitch_feature_level_overlap(
        f0_tiles, starts_y, starts_x, H, W, patch_size, overlap, downscale=4
    )
    del f0_tiles
    f1_full = _stitch_feature_level_overlap(
        f1_tiles, starts_y, starts_x, H, W, patch_size, overlap, downscale=8
    )
    del f1_tiles
    f2_full = _stitch_feature_level_overlap(
        f2_tiles, starts_y, starts_x, H, W, patch_size, overlap, downscale=16
    )
    del f2_tiles

    H0, W0 = f0_full.shape[-2:]
    # PyTorch CPU requires float32 for bilinear interpolation, cast briefly then back to half
    f1_up = (
        F.interpolate(
            f1_full.unsqueeze(0).to(torch.float32),
            size=(H0, W0),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .half()
    )
    del f1_full  # Clean array from memory immediately

    f2_up = (
        F.interpolate(
            f2_full.unsqueeze(0).to(torch.float32),
            size=(H0, W0),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .half()
    )
    del f2_full  # Clean array from memory immediately

    # print(
    #     f"Stitch features: f0 {tuple(f0_full.shape)}, f1 {tuple(f1_up.shape)}, f2 {tuple(f2_up.shape)}"
    # )
    res = torch.cat([f0_full, f1_up, f2_up], dim=0)
    del f0_full, f1_up, f2_up
    return res
