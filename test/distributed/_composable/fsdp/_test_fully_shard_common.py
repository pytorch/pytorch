# Owner(s): ["oncall: distributed"]

import contextlib
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed._composable.fsdp._fsdp_param_group import (
    FSDPParamGroup,
    RegisterPostBackwardHook,
)
from torch.distributed._tensor import DTensor


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


@contextlib.contextmanager
def patch_register_post_backward_hook_backward(new_backward: Callable):
    orig_backward = RegisterPostBackwardHook.backward
    RegisterPostBackwardHook.backward = new_backward
    try:
        yield
    finally:
        RegisterPostBackwardHook.backward = orig_backward


def reduce_scatter_with_assert(
    cls,
    orig_reduce_scatter: Callable,
    assert_fn: Callable,  # `assert_fn(output: Tensor)`
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
):
    if len(args) > 0:
        output = args[0]
    elif "output" in kwargs:
        output = kwargs["output"]
    else:
        raise AssertionError(
            f"Cannot get reduce-scatter output from\nargs: {args}\nkwargs: {kwargs}"
        )
    assert_fn(output)
    return orig_reduce_scatter(*args, **kwargs)


def check_1d_sharded_parity(
    cls,  # unit test class
    replicated_module: nn.Module,
    sharded_module: nn.Module,
    group: Optional[dist.ProcessGroup] = None,
    check_grads: bool = True,
):
    group = group or dist.distributed_c10d._get_default_group()
    rank, world_size = group.rank(), group.size()
    for (replicated_name, replicated_param), (sharded_name, sharded_param) in zip(
        replicated_module.named_parameters(), sharded_module.named_parameters()
    ):
        cls.assertEqual(replicated_name, sharded_name)
        cls.assertIsInstance(sharded_param, DTensor)
        param_chunks = torch.chunk(replicated_param, world_size, dim=0)
        cls.assertEqual(sharded_param._local_tensor, param_chunks[rank])
        if not check_grads:
            continue
        if replicated_param.grad is None:
            cls.assertIsNone(sharded_param.grad)
            continue
        cls.assertIsNotNone(sharded_param.grad)
        grad_chunks = torch.chunk(replicated_param.grad, world_size, dim=0)
        cls.assertEqual(sharded_param.grad._local_tensor, grad_chunks[rank])
