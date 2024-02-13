import torch
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union, cast

from torch.sparse.semi_structured import (
    SparseSemiStructuredTensor,
    SparseSemiStructuredTensorCUSPARSELT,
    SparseSemiStructuredTensorCUTLASS,
    to_sparse_semi_structured
)

class _Sparsify24Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, algo: str):  # type: ignore[override]
        if not isinstance(x, SparseSemiStructuredTensor):
            out = SparseSemiStructuredTensorCUSPARSELT.from_dense_fast(x, algo=algo)
        else:
            if x.threads_masks is None:
                raise ValueError("!!")
            out = x
        ctx.threads_masks = out.threads_masks
        ctx.dtype = out.dtype
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        if isinstance(grad_out, SparseSemiStructuredTensor):
            return grad_out, None, None, None
        assert not isinstance(grad_out, SparseSemiStructuredTensor)
        assert grad_out.dtype == ctx.dtype

        assert ctx.threads_masks.is_contiguous()
        grad_in = torch.ops.sparse.sparse24_apply_dense_output(grad_out, ctx.threads_masks)

        return (
            grad_in,
            None,
            None,
            None,
        )

class _Sparsify24LikeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, pattern: SparseSemiStructuredTensorCUTLASS, out_dense: bool):  # type: ignore[override]
        assert isinstance(pattern, SparseSemiStructuredTensor)
        if not isinstance(pattern, SparseSemiStructuredTensorCUTLASS):
            raise NotImplementedError(
                "`sparsify24_like(x, pattern)` is only implemented for CUTLASS backend"
            )
        if not pattern.threads_masks.is_contiguous():
            raise NotImplementedError(
                "`sparsify24_like(x, pattern)` is not implemented when `pattern` is transposed"
            )
        ctx.threads_masks = pattern.threads_masks
        ctx.meta = pattern.meta
        ctx.meta_t = pattern.meta_t
        ctx.dtype = pattern.dtype
        if out_dense:
            assert ctx.threads_masks.is_contiguous()
            return (x, ctx.threads_masks)
        packed, packed_t = torch.ops.sparse.sparse24_apply(x, ctx.threads_masks)
        return SparseSemiStructuredTensorCUTLASS(
            x.shape,
            packed,
            ctx.meta,
            packed_t,
            ctx.meta_t,
            ctx.threads_masks,
            requires_grad=x.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        if isinstance(grad_out, SparseSemiStructuredTensor):
            return grad_out, None, None
        assert not isinstance(grad_out, SparseSemiStructuredTensor)
        assert grad_out.dtype == ctx.dtype
        packed, packed_t = torch.ops.sparse.sparse24_apply(grad_out, ctx.threads_masks)
        return (
            SparseSemiStructuredTensorCUTLASS(
                grad_out.shape,
                packed,
                ctx.meta,
                packed_t,
                ctx.meta_t,
                ctx.threads_masks,
                requires_grad=grad_out.requires_grad,
            ),
            None,
            None,
        )



# We want to use `torch._dynamo.allow_in_graph` as a decorator
# (see https://fburl.com/workplace/uimiz0mf) but it breaks mypy.
# This is a hack to work around this
F = TypeVar("F", bound=Callable[..., Any])


def allow_in_graph(func: F) -> F:
    return cast(F, torch._dynamo.allow_in_graph(func))


@allow_in_graph
def sparsify24(
    x: torch.Tensor,
    algo: str = "",
    backend:str = "cutlass",
) -> SparseSemiStructuredTensor:
    return _Sparsify24Func.apply(x, algo)


@allow_in_graph
def sparsify24_like(
    x: torch.Tensor,
    pattern:
    torch.Tensor,
    out_dense: bool = False
) -> SparseSemiStructuredTensor:
    if not isinstance(pattern, SparseSemiStructuredTensor):
        raise ValueError(
            f"`pattern` must be a `SparseSemiStructuredTensor` but got a {type(pattern)}"
        )
    # Handle transposed case
    if not pattern.threads_masks.is_contiguous():
        return _Sparsify24LikeFunc.apply(x.t(), pattern.t(), out_dense).t()
    return _Sparsify24LikeFunc.apply(x, pattern, out_dense)
