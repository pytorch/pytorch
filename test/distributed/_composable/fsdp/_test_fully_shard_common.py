# Owner(s): ["oncall: distributed"]

import contextlib
from typing import Callable

import torch
import torch.distributed as dist
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


@contextlib.contextmanager
def patch_all_gather(new_all_gather_into_tensor: Callable):
    orig_all_gather = dist.all_gather_into_tensor
    dist.all_gather_into_tensor = new_all_gather_into_tensor
    try:
        yield
    finally:
        dist.all_gather_into_tensor = orig_all_gather


@contextlib.contextmanager
def patch_reduce_scatter(new_reduce_scatter_tensor: Callable):
    orig_reduce_scatter = dist.reduce_scatter_tensor
    dist.reduce_scatter_tensor = new_reduce_scatter_tensor
    try:
        yield
    finally:
        dist.reduce_scatter_tensor = orig_reduce_scatter
