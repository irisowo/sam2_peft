import cv2
import math
from typing import Iterable, Literal, Optional, Union, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================================================
class FeatureBank:

    def __init__(
        self,
        train_embeds: Iterable[Union[list, np.ndarray, torch.Tensor]],
        device: Union[str, torch.device] = "cpu",
        eps: float = 1e-8,
    ):

        self.device = torch.device(device)

        # bank = train_embeds
        bank = []
        for idx, e in enumerate(train_embeds):
            t = self._to_tensor(e)
            # t.shape=[C, H, W]
            if t.dim() == 4 and t.size(0) == 1:
                t = t.squeeze(0)
            bank.append(t.to(self.device, dtype=torch.float32))

        # Asserttion : t.shape==[C, H, W] for t in self.bank
        c, h, w = bank[0].shape
        self._assert_chw(bank, c, h, w)

        # self.bank.shape=[N, C, H, W]
        self.bank = torch.stack(bank, dim=0).contiguous() if len(
            bank) > 1 else bank[0].unsqueeze(0).contiguous()
        self.N, self.C, self.H, self.W = self.bank.shape

        # mu/var 形狀: [C,H,W]
        self.mu = self.bank.mean(dim=0)  # [C,H,W]
        self.var = self.bank.var(dim=0, unbiased=False) + eps  # [C,H,W]
        self.inv_std = self.var.sqrt().reciprocal()  # [C,H,W]
        self.inv_var = self.var.reciprocal()  # [C,H,W] 供 Mahalanobis

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
                c, h, w), f"Mismatch t[{i}]shape={t.shape} vs. {(c, h, w)}"

    # @torch.no_grad()
    # def cal_anomaly_map(
    #     self,
    #     test_embed: Union[np.ndarray, torch.Tensor],
    #     k: int = 4,  # 新增參數 k
    # ) -> np.ndarray:
    #     t = self._to_tensor(test_embed)
    #     if t.dim() == 4 and t.size(0) == 1:
    #         t = t.squeeze(0)

    #     self._assert_chw([t], self.C, self.H, self.W)

    #     t = t.to(self.device, dtype=torch.float32).unsqueeze(0)
    #     b = self.bank # [n, C, H, W]

    #     # 1. 計算餘弦相似度並轉為歐氏距離
    #     t_norm = F.normalize(t, p=2, dim=1)  # [1, C, H, W]
    #     b_norm = F.normalize(b, p=2, dim=1)  # [n, C, H, W]
        
    #     # 矩陣乘法優化：計算所有 bank 樣本與測試樣本的相似度
    #     # 產出形狀 [n, H, W]
    #     sim = (t_norm * b_norm).sum(dim=1) 
        
    #     # 轉為距離矩陣 [n, H, W]
    #     dists = (2 * (1.0 - sim)).clamp_min(0.0).sqrt()
    #     padding = 1
    #     dists_smoothed = F.avg_pool2d(
    #         dists, kernel_size=3, stride=1, padding=padding
    #     ) # [n, H, W]

    #     # 再從平滑後的距離中找最小值
    #     n, H, W = dists_smoothed.shape
    #     d_hw = dists_smoothed.view(n, -1)
    #     anomaly, _ = d_hw.min(dim=0)

    #     # 4. 後處理：回復形狀並轉回 Numpy
    #     anomaly = anomaly.view(H, W).detach().cpu().numpy()
        
    #     return anomaly

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
        t = t.to(self.device, dtype=torch.float32).unsqueeze(0)

        # b.shape=[n, C, H, W]
        b = self.bank

        t_norm = F.normalize(t, p=2, dim=1)  # [1, C, H, W],
        b_norm = F.normalize(b, p=2, dim=1)  # [n, C, H, W]
        sim = (t_norm * b_norm).sum(dim=1)  # [n, H, W]

        # dists = (1.0 - sim).squeeze(1)  # [n, H, W]
        # dists = 1.0 - (sim + 1.0) / 2.0  # [n, H, W], turn to [0,1] range
        # dists = dists.pow(0.5)
        dists = 2 * (1.0 - sim)
        dists = dists.clamp_min(0.0).sqrt()  # [n, H, W], Euclidean from cosine sim
        # ==========================================================
        # 關鍵修改：加入平滑化 (Smoothing) 以消除小白點
        # ==========================================================
        # 使用 Average Pooling 模擬鄰域特徵的聚合，降低對微小位移的敏感度
        kernel_size = 3  # 可以嘗試 3, 5, 或 7
        padding = kernel_size // 2
        
        # 注意：我們希望對每個 bank sample 的距離圖都做平滑
        # dists 形狀為 [n, H, W]，需要增加 channel 維度才能進 conv2d
        dists = dists.unsqueeze(1) # [n, 1, H, W]
        dists = F.avg_pool2d(dists, kernel_size=kernel_size, stride=1, padding=padding)
        dists = dists.squeeze(1)   # [n, H, W]

        # [n, H, W] --> [n, HW] --> [HW]
        n, H, W = dists.shape
        d_hw = dists.view(n, -1)

        anomaly, _ = d_hw.min(dim=0)  # [HW]

        # [HW] --> [H, W]
        anomaly = anomaly.view(H, W).detach().cpu().numpy()
        # print(f'max val: {anomaly.max()}, min val: {anomaly.min()}')
        return anomaly


# =============================================================================================================
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
            (L - patch_size) / stride)  # how many stride moves after first
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
    assert t0.ndim == 4 and t0.shape[
        0] == 1, f"expect [1,C,h,w], got {tuple(t0.shape)}"
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
            acc[:, yf:yf + ht, xf:xf + wt] += tile
            cnt[:, yf:yf + ht, xf:xf + wt] += 1.0
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
            patch = padded[y:y + patch_size, x:x + patch_size, :]
            predictor.set_image(patch)
            f0, f1, f2 = predictor.get_feat_list()  # [1,C,h,w]
            f0_tiles.append(f0.detach())
            f1_tiles.append(f1.detach())
            f2_tiles.append(f2.detach())

    f0_full = _stitch_feature_level_overlap(f0_tiles,
                                            starts_y,
                                            starts_x,
                                            H,
                                            W,
                                            patch_size,
                                            overlap,
                                            downscale=4)
    f1_full = _stitch_feature_level_overlap(f1_tiles,
                                            starts_y,
                                            starts_x,
                                            H,
                                            W,
                                            patch_size,
                                            overlap,
                                            downscale=8)
    f2_full = _stitch_feature_level_overlap(f2_tiles,
                                            starts_y,
                                            starts_x,
                                            H,
                                            W,
                                            patch_size,
                                            overlap,
                                            downscale=16)

    H0, W0 = f0_full.shape[-2:]
    f1_up = F.interpolate(f1_full.unsqueeze(0),
                          size=(H0, W0),
                          mode="bilinear",
                          align_corners=False).squeeze(0)
    f2_up = F.interpolate(f2_full.unsqueeze(0),
                          size=(H0, W0),
                          mode="bilinear",
                          align_corners=False).squeeze(0)

    # print(
    #     f"Stitch features: f0 {tuple(f0_full.shape)}, f1 {tuple(f1_up.shape)}, f2 {tuple(f2_up.shape)}"
    # )
    return torch.cat([f0_full, f1_up, f2_up], dim=0)


# =============================================================================================================
# def _stitch_feature_level(feature_tiles, H, W, patch_size, downscale):
#     """
#     feature_tiles: 依 row-major 順序的 list[Tensor]，每個 Tensor 形狀 [1, C, h_l, w_l]
#     H, W: 原圖高寬（patch 前）
#     patch_size: 影像 patch 邊長（輸入空間）
#     downscale: 該特徵相對輸入空間的縮小倍率（例如 f0=4, f1=8, f2=16）
#     return: Tensor [C, H_l, W_l]，其中 H_l = ceil(H/downscale)
#     """
#     device = feature_tiles[0].device
#     C = feature_tiles[0].shape[1]

#     full_H = math.ceil(H / patch_size) * patch_size
#     full_W = math.ceil(W / patch_size) * patch_size
#     full_H_l = full_H // downscale
#     full_W_l = full_W // downscale
#     H_l = math.ceil(H / downscale)
#     W_l = math.ceil(W / downscale)

#     canvas = torch.zeros((C, full_H_l, full_W_l), device=device)
#     counter = torch.zeros((1, full_H_l, full_W_l), device=device)

#     idx = 0
#     for y in range(0, full_H, patch_size):
#         for x in range(0, full_W, patch_size):
#             f = feature_tiles[idx].squeeze(0)  # [C, h_l, w_l]
#             h_l, w_l = f.shape[-2:]
#             y_l = y // downscale
#             x_l = x // downscale
#             canvas[:, y_l:y_l + h_l, x_l:x_l + w_l] += f
#             counter[:, y_l:y_l + h_l, x_l:x_l + w_l] += 1
#             idx += 1

#     canvas = canvas / counter
#     canvas = canvas[:, :H_l, :W_l]  # 去掉 padding
#     return canvas  # [C, H_l, W_l]

# @torch.no_grad()
# def stitch_sam2_patch_feats(predictor, full_image, patch_size=1024):
#     """
#     full_image : np.ndarray HxWxC RGB
#     patch_size: P, default=1024
#     假設 model._features 有：
#       - "high_res_feats"[0] -> f0（≈ 1/4）
#       - "high_res_feats"[1] -> f1（≈ 1/8）
#       - "image_embed"       -> f2（≈ 1/16）
#     return: Tensor [C_total, H0, W0]，H0 = ceil(H/4), W0 = ceil(W/4)
#     """
#     # to_tensor:
#     H, W, C = full_image.shape
#     inp_patches = []
#     for y in range(0, H, patch_size):
#         for x in range(0, W, patch_size):
#             patch = full_image[y:y + patch_size, x:x + patch_size, :]
#             ph = min(patch_size, H - y)
#             pw = min(patch_size, W - x)
#             if ph < patch_size or pw < patch_size:
#                 pad_bottom = patch_size - ph
#                 pad_right = patch_size - pw
#                 patch = np.pad(
#                     patch,
#                     ((0, pad_bottom), (0, pad_right), (0, 0)),
#                     mode='constant',
#                     constant_values=0,
#                 )
#             inp_patches.append(patch)

#     f0_tiles, f1_tiles, f2_tiles = [], [], []
#     for patch in inp_patches:
#         predictor.set_image(patch)
#         f0, f1, f2 = predictor.get_feat_list()
#         f0_tiles.append(f0.detach())
#         f1_tiles.append(f1.detach())
#         f2_tiles.append(f2.detach())

#     f0_full = _stitch_feature_level(f0_tiles, H, W, patch_size, downscale=4)
#     f1_full = _stitch_feature_level(f1_tiles, H, W, patch_size, downscale=8)
#     f2_full = _stitch_feature_level(f2_tiles, H, W, patch_size, downscale=16)

#     # 將 f1、f2 上採樣到 f0 的空間，再做通道串接
#     H0, W0 = f0_full.shape[-2], f0_full.shape[-1]
#     f1_up = F.interpolate(f1_full.unsqueeze(0),
#                           size=(H0, W0),
#                           mode='bilinear',
#                           align_corners=False).squeeze(0)
#     f2_up = F.interpolate(f2_full.unsqueeze(0),
#                           size=(H0, W0),
#                           mode='bilinear',
#                           align_corners=False).squeeze(0)
#     print(
#         f'  Stitch features: f0 {f0_full.shape}, f1 {f1_up.shape}, f2 {f2_up.shape}'
#     )
#     image_embedding = torch.cat([f0_full, f1_up, f2_up],
#                                 dim=0)  # [C_total, H0, W0]
#     return image_embedding
