# Owner(s): ["oncall: distributed"]

import contextlib
from typing import Callable, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup


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


class DoubleLinear(nn.Module):
    """
    This can be used for returning multiple outputs from a module
    (``use_second_linear=True``) or for having an unused module (``False``).
    """

    def __init__(self, dim: int, use_second_linear: bool = True):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.use_second_linear = use_second_linear

    def forward(
        self, x: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.use_second_linear:
            return self.relu(self.lin1(x)), self.relu(self.lin2(x))
        return self.relu(self.lin1(x))


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


@contextlib.contextmanager
def patch_unshard(new_unshard: Callable):
    orig_unshard = FSDPParamGroup.unshard
    FSDPParamGroup.unshard = new_unshard
    try:
        yield
    finally:
        FSDPParamGroup.unshard = orig_unshard


@contextlib.contextmanager
def patch_post_backward(new_post_backward: Callable):
    orig_post_backward = FSDPParamGroup._post_backward
    FSDPParamGroup._post_backward = new_post_backward
    try:
        yield
    finally:
        FSDPParamGroup._post_backward = orig_post_backward
