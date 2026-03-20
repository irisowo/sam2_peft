# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
from typing import List, Tuple, Union
from itertools import repeat
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP

from sam2.pmodel.lora import LoRALinear, DoRALinear, RLRRLinear, EDoRALinear, DoRA
from sam2.pmodel.adapter_vpt import PromptGenerator
from sam2.pmodel.adapter import AdapterPlus


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
        # -------- PEFT --------
        peft_linear_mode: str = "none",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool

        # base linear
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

        # -------- PEFT --------
        assert peft_linear_mode in {"none", "lora", "dora", "rlrr", "edora"}
        self.peft_linear_mode = peft_linear_mode  # "none" / "lora" / "dora" / "rlrr"
        

        # -------- PEFT --------
        self.qkv_delta, self.proj_delta = self._build_peft_linear(
            peft_linear_mode, lora_r, lora_alpha, lora_dropout)

    # ---- PEFT  Injection ----
    def _build_peft_linear(
        self,
        peft_linear_mode: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        # -------- PEFT --------
        if peft_linear_mode == "lora":
            qkv_delta = LoRALinear(self.qkv, lora_r, lora_alpha, lora_dropout)
            proj_delta = LoRALinear(self.proj, lora_r, lora_alpha,
                                    lora_dropout)
        elif peft_linear_mode == "dora":
            qkv_delta = DoRALinear(self.qkv, lora_r, lora_alpha, lora_dropout)
            proj_delta = DoRALinear(self.proj, lora_r, lora_alpha,
                                    lora_dropout)
        elif peft_linear_mode == "edora":
            qkv_delta = EDoRALinear(self.qkv, lora_r, init_std=0.01)
            proj_delta = EDoRALinear(self.proj, lora_r, init_std=0.01)
        elif peft_linear_mode == "rlrr":
            qkv_delta = RLRRLinear(self.qkv)
            proj_delta = RLRRLinear(self.proj)
        else:
            qkv_delta = None
            proj_delta = None
        return qkv_delta, proj_delta

    # ---- PEFT  Injection ----
    def _apply_qkv(self, x: torch.Tensor) -> torch.Tensor:
        if self.qkv_delta is None:
            return self.qkv(x)
        return self.qkv_delta(x)

    # ---- PEFT  Injection ----
    def _apply_proj(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj_delta is None:
            return self.proj(x)
        return self.proj_delta(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        # ---- PEFT  Injection ----
        qkv = self._apply_qkv(x)

        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        # ---- PEFT  Injection ----
        x_out = self._apply_proj(x)

        return x_out


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        # -------- PEFT --------
        peft_linear_mode: str = "none",
        adapter_mode: str = "none",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        adapter_plus_rank: int = 8,
        peft_target: str = "attn",  # 'none', 'attn', 'all'
        # -------- PEFT --------
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        # 1. attn_peft_mode
        attn_peft_mode = peft_linear_mode if peft_target in ["attn", "all"
                                                             ] else "none"
        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
            # -------- PEFT --------
            peft_linear_mode=attn_peft_mode,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # -------- PEFT --------
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )
        # 2. mlp_peft_mode
        if peft_target in ["mlp", "all"] and peft_linear_mode != "none":
            self._inject_peft_to_module(self.mlp, peft_linear_mode, lora_r,
                                        lora_alpha, lora_dropout)
        # -------- PEFT --------
        self.adapter_mode = adapter_mode
        if adapter_mode == "adapter_plus":
            self.adapter_plus = AdapterPlus(
                dim=dim_out,
                rank=adapter_plus_rank,
                bias=True,
                dropout=0.0,
                drop_path=0.1,
            )
        # -------- PEFT --------

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def _inject_peft_to_module(self, module, mode, r, alpha, dropout):
        """
            Recursively traverse the module and replace nn.Linear with PEFT variants.
            """
        # Iterate over named children to allow in-place replacement
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Build the PEFT layer wrapper
                if mode == "dora":
                    new_layer = DoRA(child, r, alpha, dropout)
                else:
                    continue  # Should not happen if check is done before calling
                # Replace the layer
                setattr(module, name, new_layer)
                print(f"Injected {mode} into {name} of MLP")
            else:
                # Recursively check sub-modules (e.g. if MLP contains Sequential)
                self._inject_peft_to_module(child, mode, r, alpha, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # -------- PEFT --------
        if self.adapter_mode == "adapter_plus":
            x = self.adapter_plus(x)
        # -------- PEFT --------
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 144,  # initial embed dim
        num_heads: int = 2,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 6, 36, 4),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (7, 7),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            16,
            8,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            23,
            33,
            43,
        ),
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
        # -------- PEFT --------
        prompt_mode: str = "none",  # "none" / "prompt_generator"
        peft_linear_mode: str = "none",  # "none" / "lora" / "dora" / "rlrr"
        peft_target: str = "attn",  # 'none', 'attn', 'mlp', 'all'
        adapter_mode: str = "none",  # "none" / "adapter_plus"
        lora_r: int = 8,
        adapter_plus_rank: int = 8,
        # -------- PEFT --------
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        # -------- PEFT --------
        self.prompt_mode = prompt_mode
        print(f'[Hiera] prompt_mode: {prompt_mode}\n' + \
            f'peft_linear_mode: {peft_linear_mode}, lora_r: {lora_r}\n' + \
            f'adapter_mode: {adapter_mode}, adapter_plus_rank: {adapter_plus_rank}'
        )
        assert prompt_mode in {"none", "prompt_generator"}
        assert peft_linear_mode in {"none", "lora", "dora", "rlrr", "edora"}
        assert adapter_mode in {"none", "adapter_plus"}
        # -------- PEFT --------

        cur_stage = 1
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                # -------- PEFT --------
                peft_linear_mode=peft_linear_mode,
                peft_target=peft_target,
                adapter_mode=adapter_mode,
                lora_r=lora_r,
                lora_alpha=16,
                lora_dropout=0.1,
                adapter_plus_rank=adapter_plus_rank,
                # -------- PEFT --------
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

        # -------- PEFT --------
        # [C1, C2, C3, C4] = [144, 288, 576, 1152]
        stage_dims = [
            self.blocks[self.stage_ends[i]].dim_out for i in range(4)
        ]
        if self.prompt_mode == "prompt_generator":
            self.prompt_generator = PromptGenerator(
                embed_dims=stage_dims,
                depths=stages,
                img_size=1024,
            )

    # -------- PEFT --------
        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    # -------- PEFT --------
    def _init_prompt_state(self, raw: torch.Tensor, x: torch.Tensor):
        if self.prompt_mode != "prompt_generator":
            return None

        hc1, hc2, hc3, hc4 = self.prompt_generator.init_handcrafted(raw)
        return {1: hc1, 2: hc2, 3: hc3, 4: hc4}

    def _get_stage_prompt(self, x: torch.Tensor, stage_hc, stage_idx: int):
        if self.prompt_mode != "prompt_generator":
            return None
        # init_prompt returns (handcrafted, embedding)
        hc = stage_hc[stage_idx]
        return self.prompt_generator.init_prompt(x, hc, stage_idx)

    def _apply_prompt_to_block(
        self,
        x: torch.Tensor,
        cur_prompt,
        stage_idx: int,
        block_idx_in_stage: int,
    ) -> torch.Tensor:
        if self.prompt_mode != "prompt_generator":
            return x
        if cur_prompt is None:
            return x
        return self.prompt_generator.add_prompt(x, cur_prompt, stage_idx,
                                                block_idx_in_stage)

    # -------- PEFT --------

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        raw = x
        x = self.patch_embed(x)  # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        # prepare handcrafted prompts once per image
        stage_hc = self._init_prompt_state(raw, x)

        outputs = []
        stage_idx = 1
        block_idx_in_stage = 0

        # -------- PEFT --------
        cur_prompt = None
        if stage_hc is not None:
            cur_prompt = self._get_stage_prompt(x, stage_hc, stage_idx)
        # -------- PEFT --------

        for i, blk in enumerate(self.blocks):
            # -------- PEFT --------
            x = self._apply_prompt_to_block(x, cur_prompt, stage_idx,
                                            block_idx_in_stage)
            # -------- PEFT --------

            x = blk(x)

            # self.stage_ends=[1, 7, 43, 47]
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

            # stage bookkeeping
            block_idx_in_stage += 1

            # q_pool_blocks=[2, 8, 44]
            if i in self.q_pool_blocks:
                stage_idx += 1
                block_idx_in_stage = 0

                # -------- PEFT --------
                if stage_hc is not None:
                    cur_prompt = self._get_stage_prompt(x, stage_hc, stage_idx)
                # -------- PEFT --------

        return outputs

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)
