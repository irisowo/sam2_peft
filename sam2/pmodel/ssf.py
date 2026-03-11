import torch
import torch.nn as nn

class SSF(nn.Module):
    """
    Scale-Shift Feature (SSF)
    out = x * (1 + scale) + shift
    """
    def __init__(self, dim, prefix="ssf"):
        super().__init__()
        self.ssf_scale = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.ssf_shift = nn.Parameter(torch.zeros(dim), requires_grad=True)

        # rename parameters to support freeze-by-name
        self.ssf_scale.name = f"{prefix}_scale"
        self.ssf_shift.name = f"{prefix}_shift"

    def forward(self, x):
        return x * (1.0 + self.ssf_scale) + self.ssf_shift
