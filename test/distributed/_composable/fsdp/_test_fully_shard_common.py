# Owner(s): ["oncall: distributed"]


import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        device: torch.device = torch.device("cpu"),
        with_buffer: bool = False,
        dim_multiplier: int = 4,
    ):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim_multiplier * dim, device=device)
        self.out_proj = nn.Linear(dim_multiplier * dim, dim, device=device)
        if with_buffer:
            self.register_buffer("buffer", torch.randn((dim,), device=device))
        else:
            self.buffer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z = F.relu(z)
        z = self.out_proj(z)
        z = F.relu(z)
        if self.buffer:
            z += self.buffer
        return z
