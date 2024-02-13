import torch
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union, cast

from torch.sparse.semi_structured import (
    SparseSemiStructuredTensor,
    SparseSemiStructuredTensorCUSPARSELT,
    SparseSemiStructuredTensorCUTLASS,
    to_sparse_semi_structured
)

class _Sparsify24LikeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, pattern: SparseSemiStructuredTensor, out_dense: bool):  # type: ignore[override]
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
            return torch.ops.sparse.sparse24_apply_dense_output(x, ctx.threads_masks)

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

class _Sparsify24Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, algo: str, gradient: str, backend: str):  # type: ignore[override]
        if not isinstance(x, SparseSemiStructuredTensor):
            (packed, meta, packed_t, meta_t, threads_masks) = torch.ops.sparse.sparse24_sparsify_both_ways(
                x, algorithm=algo, backend=backend
            )
            cls = (
                SparseSemiStructuredTensorCUTLASS
                if backend == "cutlass"
                else SparseSemiStructuredTensorCUSPARSELT
            )
            out = cls(
                x.shape,
                packed=packed,
                meta=meta,
                packed_t=packed_t,
                meta_t=meta_t,
                threads_masks=threads_masks,
                requires_grad=False,
                fuse_transpose_cusparselt=True,
            )
        else:
            if x.threads_masks is None:
                raise ValueError("!!")
            out = x
        ctx.threads_masks = out.threads_masks
        ctx.meta = out.meta
        ctx.meta_t = out.meta_t
        ctx.dtype = out.dtype
        ctx.gradient = gradient
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        if isinstance(grad_out, SparseSemiStructuredTensor):
            return (grad_out, None, None, None)

        assert not isinstance(grad_out, SparseSemiStructuredTensor)
        assert grad_out.dtype == ctx.dtype
        if ctx.gradient == "24sparse":
            packed, packed_t = torch.ops.sparse.sparse24_apply(grad_out, ctx.threads_masks)
            grad_in: torch.Tensor = SparseSemiStructuredTensorCUTLASS(
                grad_out.shape,
                packed,
                ctx.meta,
                packed_t,
                ctx.meta_t,
                ctx.threads_masks,
                requires_grad=grad_out.requires_grad,
            )
        elif ctx.gradient == "24dense":
            assert ctx.threads_masks.is_contiguous()
            grad_in = torch.ops.sparse.sparse24_apply_dense_output(grad_out, ctx.threads_masks)
        else:
            assert False, f"Unsupported gradient type: {ctx.gradient}"

        return (grad_in, None, None, None)
