import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from itertools import repeat


# ------------------------------
# Utility helpers
# ------------------------------
def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return tuple(repeat(x, 2))


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Truncated normal init (kept here to avoid extra deps)."""

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


# ------------------------------
# Extra blocks for prompt-based tuning (ported & adapted)
# ------------------------------
class OverlapPatchEmbed(nn.Module):
    """Image -> overlapped patch embedding (returns flattened tokens)."""

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PromptGenerator(nn.Module):
    """Generates stage-wise prompts from handcrafted (image-driven) and
    embedding-driven features, then adapts them to the current stage width.
    """

    def __init__(
        self,
        embed_dims: List[int],
        depths: Tuple[int, ...],
        img_size: int,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.tuning_stage = '1234'
        self.depths = [2, 6, 36, 4]  # Hiera-Large depths
        self.scale_factor = 32
        self.prompt_type = 'highpass'
        self.input_type = 'fft'
        self.freq_nums = 0.25
        self.handcrafted_tune = True
        self.embedding_tune = True
        # support both spellings: "adapter" and "adaptor"
        self.adaptor = 'adaptor'  # 'all'

        if self.input_type == "all":
            # trainable image-sized prompt (B is added in forward)
            self.prompt = nn.Parameter(torch.zeros(3, img_size, img_size),
                                       requires_grad=False)

        # Handcrafted prompt branches (from image, via overlapped patch embed)
        if self.handcrafted_tune:
            if "1" in self.tuning_stage:
                self.handcrafted_generator1 = OverlapPatchEmbed(
                    img_size=img_size,
                    patch_size=7,
                    stride=4,
                    in_chans=3,
                    embed_dim=self.embed_dims[0] // self.scale_factor)
            if "2" in self.tuning_stage:
                self.handcrafted_generator2 = OverlapPatchEmbed(
                    img_size=img_size // 4,
                    patch_size=3,
                    stride=2,
                    in_chans=self.embed_dims[0] // self.scale_factor,
                    embed_dim=self.embed_dims[1] // self.scale_factor,
                )
            if "3" in self.tuning_stage:
                self.handcrafted_generator3 = OverlapPatchEmbed(
                    img_size=img_size // 8,
                    patch_size=3,
                    stride=2,
                    in_chans=self.embed_dims[1] // self.scale_factor,
                    embed_dim=self.embed_dims[2] // self.scale_factor,
                )
            if "4" in self.tuning_stage:
                self.handcrafted_generator4 = OverlapPatchEmbed(
                    img_size=img_size // 16,
                    patch_size=3,
                    stride=2,
                    in_chans=self.embed_dims[2] // self.scale_factor,
                    embed_dim=self.embed_dims[3] // self.scale_factor,
                )

        # Embedding prompt branches (reduce channel, then later project back)
        if self.embedding_tune:
            if "1" in self.tuning_stage:
                self.embedding_generator1 = nn.Linear(
                    self.embed_dims[0],
                    self.embed_dims[0] // self.scale_factor)
            if "2" in self.tuning_stage:
                self.embedding_generator2 = nn.Linear(
                    self.embed_dims[1],
                    self.embed_dims[1] // self.scale_factor)
            if "3" in self.tuning_stage:
                self.embedding_generator3 = nn.Linear(
                    self.embed_dims[2],
                    self.embed_dims[2] // self.scale_factor)
            if "4" in self.tuning_stage:
                self.embedding_generator4 = nn.Linear(
                    self.embed_dims[3],
                    self.embed_dims[3] // self.scale_factor)

        # Lightweight adaptors per block (unshared) + shared projection back to stage width
        if self.adaptor == "adaptor":
            if "1" in self.tuning_stage:
                for i in range(self.depths[0] + 1):
                    setattr(
                        self,
                        f"lightweight_mlp1_{i}",
                        nn.Sequential(
                            nn.Linear(self.embed_dims[0] // self.scale_factor,
                                      self.embed_dims[0] // self.scale_factor),
                            nn.GELU(),
                        ),
                    )
                self.shared_mlp1 = nn.Linear(
                    self.embed_dims[0] // self.scale_factor,
                    self.embed_dims[0])
            if "2" in self.tuning_stage:
                for i in range(self.depths[1] + 1):
                    setattr(
                        self,
                        f"lightweight_mlp2_{i}",
                        nn.Sequential(
                            nn.Linear(self.embed_dims[1] // self.scale_factor,
                                      self.embed_dims[1] // self.scale_factor),
                            nn.GELU(),
                        ),
                    )
                self.shared_mlp2 = nn.Linear(
                    self.embed_dims[1] // self.scale_factor,
                    self.embed_dims[1])
            if "3" in self.tuning_stage:
                for i in range(self.depths[2] + 1):
                    setattr(
                        self,
                        f"lightweight_mlp3_{i}",
                        nn.Sequential(
                            nn.Linear(self.embed_dims[2] // self.scale_factor,
                                      self.embed_dims[2] // self.scale_factor),
                            nn.GELU(),
                        ),
                    )
                self.shared_mlp3 = nn.Linear(
                    self.embed_dims[2] // self.scale_factor,
                    self.embed_dims[2])
            if "4" in self.tuning_stage:
                for i in range(self.depths[3] + 1):
                    setattr(
                        self,
                        f"lightweight_mlp4_{i}",
                        nn.Sequential(
                            nn.Linear(self.embed_dims[3] // self.scale_factor,
                                      self.embed_dims[3] // self.scale_factor),
                            nn.GELU(),
                        ),
                    )
                self.shared_mlp4 = nn.Linear(
                    self.embed_dims[3] // self.scale_factor,
                    self.embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # ---------------- handcrafted / embedding prompt preparation ----------------
    @torch.no_grad()
    def _prep_image_prompt(self, x_img: torch.Tensor) -> torch.Tensor:
        """Return image prompt tensor depending on input_type."""
        if self.input_type == "fft":
            return self._fft_prompt(x_img, self.freq_nums, self.prompt_type)
        elif self.input_type == "all":
            return self.prompt.unsqueeze(0).to(x_img.device).repeat(
                x_img.shape[0], 1, 1, 1)
        else:
            raise ValueError(
                f"Unsupported input_type={self.input_type}; expected 'fft' or 'all'."
            )

    @torch.no_grad()
    def _fft_prompt(self, x: torch.Tensor, rate: float,
                    prompt_type: str) -> torch.Tensor:
        device = x.device
        mask = torch.zeros_like(x, device=device)
        w, h = x.shape[-2:]
        line = int((w * h * rate)**0.5 // 2)
        mask[:, :, w // 2 - line:w // 2 + line,
             h // 2 - line:h // 2 + line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        if prompt_type == "highpass":
            fft = fft * (1 - mask)
        elif prompt_type == "lowpass":
            fft = fft * mask
        fr, fi = fft.real, fft.imag
        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real
        inv = torch.abs(inv)
        return inv

    def init_handcrafted(self, x_img: torch.Tensor):
        x = self._prep_image_prompt(x_img)  # (B, 3, H, W)
        B = x.shape[0]
        hc1 = hc2 = hc3 = hc4 = None
        if "1" in self.tuning_stage:
            hc1, H1, W1 = self.handcrafted_generator1(x)
        if "2" in self.tuning_stage and hc1 is not None:
            hc2, H2, W2 = self.handcrafted_generator2(
                hc1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous())
        if "3" in self.tuning_stage and hc2 is not None:
            hc3, H3, W3 = self.handcrafted_generator3(
                hc2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous())
        if "4" in self.tuning_stage and hc3 is not None:
            hc4, H4, W4 = self.handcrafted_generator4(
                hc3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous())
        return hc1, hc2, hc3, hc4

    def init_prompt(self, embedding_feature: torch.Tensor,
                    handcrafted_feature: torch.Tensor, stage_idx: int):
        # embedding_feature: (B, H, W, C)
        emb = embedding_feature
        hc = handcrafted_feature
        if self.embedding_tune:
            emb = getattr(self, f"embedding_generator{stage_idx}")(emb)
        return hc, emb

    def add_prompt(self, x: torch.Tensor, prompt, stage_idx: int,
                   depth_idx: int):
        # x: (B, H, W, C), prompt=(hc_tokens?, emb_reduced)
        feat = 0
        B, H, W, _ = prompt[1].shape
        if self.handcrafted_tune and prompt[0] is not None:
            feat = feat + prompt[0].reshape(B, H, W, -1)
        if self.embedding_tune:
            feat = feat + prompt[1]

        if self.adaptor == "adaptor":
            lw = getattr(self, f"lightweight_mlp{stage_idx}_{depth_idx}")
            shared = getattr(self, f"shared_mlp{stage_idx}")
            feat = shared(lw(feat))
        else:
            raise ValueError(
                f"Unsupported adaptor '{self.adaptor}'. Use 'adapter'/'adaptor'."
            )

        return x + feat
