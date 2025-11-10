from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d


@torch.library.custom_op("cplib::flex_cp_allgather", mutates_args=())
def flex_cp_allgather(
    k: torch.Tensor, v: torch.Tensor, seq_dim: int, pg_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    k = k.contiguous()
    v = v.contiguous()
    k = funcol.all_gather_tensor(k, seq_dim, pg_name)
    v = funcol.all_gather_tensor(v, seq_dim, pg_name)
    if isinstance(k, funcol.AsyncCollectiveTensor):
        k = k.wait()
    if isinstance(v, funcol.AsyncCollectiveTensor):
        v = v.wait()
    return k, v


@flex_cp_allgather.register_fake
def _(
    k: torch.Tensor, v: torch.Tensor, seq_dim: int, pg_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    shape_k = list(k.shape)
    shape_v = list(v.shape)
    shape_k[seq_dim] *= c10d._get_group_size_by_name(pg_name)
    shape_v[seq_dim] *= c10d._get_group_size_by_name(pg_name)
    new_k = torch.empty(shape_k, dtype=k.dtype, device=k.device)
    new_v = torch.empty(shape_v, dtype=v.dtype, device=v.device)
    return new_k, new_v


@torch.library.custom_op("cplib::flex_cp_allgather_backward", mutates_args=())
def flex_cp_allgather_backward(
    grad_full_k: torch.Tensor,
    grad_full_v: torch.Tensor,
    seq_dim: int,
    pg_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_k = funcol.reduce_scatter_tensor(grad_full_k, "sum", seq_dim, pg_name)
    if isinstance(grad_k, funcol.AsyncCollectiveTensor):
        grad_k = grad_k.wait()
    grad_v = funcol.reduce_scatter_tensor(grad_full_v, "sum", seq_dim, pg_name)
    if isinstance(grad_v, funcol.AsyncCollectiveTensor):
        grad_v = grad_v.wait()

    return grad_k, grad_v


@flex_cp_allgather_backward.register_fake
def _(
    grad_full_k: torch.Tensor,
    grad_full_v: torch.Tensor,
    seq_dim: int,
    pg_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape_k = list(grad_full_k.shape)
    shape_v = list(grad_full_v.shape)
    shape_k[seq_dim] //= c10d._get_group_size_by_name(pg_name)
    shape_v[seq_dim] //= c10d._get_group_size_by_name(pg_name)
    new_grad_k = torch.empty(
        shape_k, dtype=grad_full_k.dtype, device=grad_full_k.device
    )
    new_grad_v = torch.empty(
        shape_v, dtype=grad_full_v.dtype, device=grad_full_v.device
    )
    return new_grad_k, new_grad_v


def _flex_cp_allgather_backward(
    ctx: Any, grad_full_k: torch.Tensor, grad_full_v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, None, None]:
    grad_k, grad_v = flex_cp_allgather_backward(
        grad_full_k, grad_full_v, ctx.seq_dim, ctx.pg_name
    )
    return grad_k, grad_v, None, None


def _flex_cp_setup_context(ctx: Any, inputs: Any, output: Any) -> None:
    _, _, ctx.seq_dim, ctx.pg_name = inputs


flex_cp_allgather.register_autograd(
    _flex_cp_allgather_backward, setup_context=_flex_cp_setup_context
)
