import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# named_params with 'lora' / 'dora' / 'rlrr'


class LoRALinear(nn.Module):

    def __init__(self, linear_layer: nn.Linear, r, alpha, dropout):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(linear_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, linear_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Wx + dWx, delta = dWx = B(Ax)
        delta = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        return self.linear(x) + delta


class DoRALinear(nn.Module):
    """
    Weight-normalized low-rank adaptation (DoRA)
    Reference: https://github.com/catid/dora
    """

    def __init__(self, linear_layer: nn.Linear, r, alpha, dropout):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # low-rank matrices (same as LoRA)
        self.dora_A = nn.Linear(linear_layer.in_features, r, bias=False)
        self.dora_B = nn.Linear(r, linear_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.dora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.dora_B.weight)

        # learnable magnitude term
        self.dora_m = nn.Parameter(
            linear_layer.weight.norm(p=2, dim=0, keepdim=True))

        # store base layer (used for shape & bias)
        self.linear = linear_layer

    def forward(self, x):
        # compute low-rank update
        delta_W = self.dora_B.weight @ self.dora_A.weight * self.scaling

        # combine base + update
        W = self.linear.weight + delta_W

        # normalize direction and apply learnable magnitude
        W_norm = W / (W.norm(p=2, dim=0, keepdim=True) + 1e-9)
        W_new = self.dora_m * W_norm

        # linear projection
        return F.linear(x, W_new, self.linear.bias)


class EDoRALinear(nn.Module):
    """
    Efficient Weight-Decomposed Low-Rank Adaptation (EDoRA)
    EDoRA decomposes pre-trained weights into magnitude and directional components:
    W' = m * (W0 + BRA) / ||W0 + BRA||_c
    where:
    - m: trainable magnitude vector (n,)
    - W0: frozen pre-trained weight (m, n)
    - B, A: frozen low-rank matrices initialized via SVD
    - R: trainable small matrix (r, r)
    Reference: https://arxiv.org/abs/2501.12067
    """

    def __init__(self,
                 linear_layer: nn.Linear,
                 r: int,
                 init_std: float = 0.01):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # Store base weight
        W0 = linear_layer.weight.data  # (m, n)
        column_norms = W0.norm(p=2, dim=0, keepdim=True)  # (1, n)

        # Step 1: Decompose W0 into magnitude and direction
        D = W0 / (column_norms + 1e-9)  # (m, n)
        self.edora_m = nn.Parameter(column_norms.squeeze(0))  # (n,)

        # Step 2: SVD initialization for A and B matrices
        # Perform SVD on directional matrix D
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)

        # Take top-r singular values and vectors
        U_r = U[:, :r]  # (m, r)
        S_r = S[:r]  # (r,)
        V_r = Vh[:r, :]  # (r, n)

        # Initialize frozen matrices A and B
        # A = V_r^T = (n, r), B = U_r @ diag(S_r) = (m, r)
        self.register_buffer('A', V_r.t().contiguous())  # (n, r)
        self.register_buffer('B',
                             (U_r * S_r.unsqueeze(0)).contiguous())  # (m, r)

        # Step 3: Initialize trainable matrix R (r, r)
        self.edora_R = nn.Parameter(torch.randn(r, r) * init_std)

        # Store whether to use bias
        self.has_bias = linear_layer.bias is not None

    def forward(self, x):
        """
        Forward pass implementing:
        W' = m * (W0 + BRA) / ||W0 + BRA||_c
        y = x @ W'^T + b
        """
        # Get base weight
        delta_W = self.B @ self.edora_R @ self.A.t()  # (m, n)

        # Updated directional component
        W = self.linear.weight + delta_W  # (m, n)

        # normalize direction and apply learnable magnitude
        W_norm = W / (W.norm(p=2, dim=0, keepdim=True) + 1e-9)  # (m, n)
        W_new = self.edora_m.unsqueeze(0) * W_norm  # (m, n)

        # Apply linear transformation
        output = F.linear(x, W_new, self.linear.bias)
        return output


class RLRRLinear(nn.Module):
    """Low-Rank Rescaled Residual Linear Layer (CVPR 2024 RLRR)"""

    # Residual-based Low-Rank Rescaling (RLRR) Linear wrapper
    # W' = (1 + s_line @ s_col) ⊙ W,  b' = b + Δb
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        self.rlrr_scale_col = nn.Parameter(torch.empty(1, self.in_features))
        self.rlrr_scale_line = nn.Parameter(torch.empty(self.out_features, 1))
        assert self.linear.bias is not None, "RLRR shift bias requires base linear to have bias."
        self.rlrr_shift_bias = (nn.Parameter(torch.empty(1, self.out_features))
                                if self.linear.bias is not None else None)
        # init
        nn.init.kaiming_uniform_(self.rlrr_scale_col, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.rlrr_scale_line, a=math.sqrt(5))
        if self.rlrr_shift_bias is not None:
            nn.init.zeros_(self.rlrr_shift_bias)

    def forward(self, x):
        # return self.linear(x)
        base_w = self.linear.weight
        # (out, in) = (out,1) @ (1,in)
        scale_mat = self.rlrr_scale_line @ self.rlrr_scale_col
        weight = base_w * (1.0 + scale_mat)
        if self.linear.bias is not None:
            bias = self.linear.bias + (self.rlrr_shift_bias if
                                       self.rlrr_shift_bias is not None else 0)
        else:
            bias = None
        return F.linear(x, weight, bias)


# ======================================================================
class DoRA(nn.Module):
    """
    Weight-normalized low-rank adaptation (DoRA)
    """

    def __init__(self, linear_layer: nn.Linear, r, alpha, dropout):
        super().__init__()
        # [Modify] 直接繼承參數，解決 key mismatch
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # low-rank matrices
        self.dora_A = nn.Linear(linear_layer.in_features, r, bias=False)
        self.dora_B = nn.Linear(r, linear_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.dora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.dora_B.weight)

        # learnable magnitude term
        self.dora_m = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x):
        # compute low-rank update
        delta_W = self.dora_B.weight @ self.dora_A.weight * self.scaling

        # combine base + update
        W = self.weight + delta_W

        # normalize direction and apply learnable magnitude
        W_norm = W / (W.norm(p=2, dim=0, keepdim=True) + 1e-9)
        W_new = self.dora_m * W_norm

        # linear projection
        return F.linear(x, W_new, self.bias)
