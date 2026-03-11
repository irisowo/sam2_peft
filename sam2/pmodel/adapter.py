import torch
import torch.nn as nn
import torch.nn.functional as F

# pmodel/adapter.py

import torch
import torch.nn as nn
from sam2.modeling.sam2_utils import DropPath


class AdapterPlus(nn.Module):
    """
    Adapter+ (Steitz & Roth, CVPR 2024)
    - bottleneck rank: dim -> r -> dim
    - Setting : init=Houlsby initialization, placement=Post-Adapter, no layer norm
    - https://github.com/visinf/adapter_plus/blob/main/adapter_plus/vit_adapter.py
    """

    def __init__(
        self,
        dim: int,
        rank: int = 8,
        bias: bool = True,
        dropout: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()

        self.dim = dim
        self.rank = rank

        self.down_proj = nn.Linear(dim, rank, bias=bias)
        self.up_proj = nn.Linear(rank, dim, bias=bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # channel-wise scale
        self.scale = nn.Parameter(torch.ones(dim))

        # Adapter 自己的 DropPath（對應 config.drop_path）
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Houlsby initialization
        std = 0.02
        nn.init.trunc_normal_(self.down_proj.weight, std=std)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

        nn.init.trunc_normal_(self.up_proj.weight, std=std)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape [..., C]， [B, H, W, C] or [B, N, C]
        y = x + DropPath(scale * FF(x))
        """
        residual = x

        # x = self.norm(x)
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        x = self.dropout(x)

        if self.scale is not None:
            x = x * self.scale

        x = self.drop_path(x)
        return residual + x
