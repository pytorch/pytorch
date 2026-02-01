# mypy: allow-untyped-defs
import contextlib

import torch


__all__ = [
    "fallback_dispatcher",
    "semi_sparse_values",
    "semi_sparse_indices",
    "semi_sparse_t",
    "semi_sparse_view",
    "semi_sparse_detach",
    "semi_sparse_mm",
    "semi_sparse_addmm",
    "semi_sparse_linear",
    "semi_sparse_scaled_mm",
]


@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def fallback_dispatcher(func, types, args, kwargs):
    with no_dispatch():
        return func(*args)


def semi_sparse_values(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")
    A = args[0]
    if not isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(A).__name__}"
        )
    if A.packed is None:
        raise AssertionError("A.packed must not be None")
    if A.meta is None:
        m, k = A.shape
        num_kept_elements = m * k // 2
        return A.packed.ravel()[:num_kept_elements:].view(m, -1)
    else:
        return A.packed.detach()


def semi_sparse_indices(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")
    A = args[0]
    if not isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(A).__name__}"
        )
    if A.packed is None:
        raise AssertionError("A.packed must not be None")
    if A.meta is None:
        m, k = A.shape
        num_kept_elements = m * k // 2
        metadata = A.packed.ravel()[num_kept_elements:].view(m, -1)
        return metadata.view(torch.int32 if A.dtype == torch.int32 else torch.int16)
    else:
        return A.meta


def semi_sparse_t(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")
    self = args[0]
    if not isinstance(self, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(self).__name__}"
        )
    if len(self.shape) != 2:
        raise AssertionError(f"expected 2D tensor, got {len(self.shape)}D")
    # Because we cannot go from the compressed representation back to the dense representation currently,
    # we just keep track of how many times we have been transposed. Depending on whether the sparse matrix
    # is the first or second argument, we expect an even / odd number of calls to transpose respectively.
    # pyrefly: ignore [no-matching-overload]
    return self.__class__(
        torch.Size([self.shape[-1], self.shape[0]]),
        packed=self.packed_t,
        meta=self.meta_t,
        packed_t=self.packed,
        meta_t=self.meta,
        compressed_swizzled_bitmask=(
            self.compressed_swizzled_bitmask.transpose(0, 1)
            if self.compressed_swizzled_bitmask is not None
            else None
        ),
        fuse_transpose_cusparselt=args[0].fuse_transpose_cusparselt,
        alg_id_cusparselt=args[0].alg_id_cusparselt,
    )


def semi_sparse_view(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) != 2:
        raise AssertionError(f"expected 2 args, got {len(args)}")
    self, shape = args
    if tuple(shape) != self.shape:
        raise NotImplementedError(
            f"`view` is not implemented for SparseSemiStructuredTensor, except for the dummy case (shape={shape})"
        )
    return self


def semi_sparse_detach(func, types, args, kwargs) -> torch.Tensor:
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")
    self = args[0]
    return self.__class__(
        shape=self.shape,
        packed=self.packed,
        meta=self.meta,
        packed_t=self.packed_t,
        meta_t=self.meta_t,
        compressed_swizzled_bitmask=self.compressed_swizzled_bitmask,
        fuse_transpose_cusparselt=self.fuse_transpose_cusparselt,
        alg_id_cusparselt=self.alg_id_cusparselt,
        requires_grad=False,
    )


def semi_sparse_mm(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) != 2:
        raise AssertionError(f"expected 2 args, got {len(args)}")
    A, B = args
    if A.ndim != 2 or B.ndim != 2:
        raise NotImplementedError(
            "`SparseSemiStructuredTensor` matmul: Broadcasting is not implemented"
        )
    if isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        row, col = B.shape
        B_padded = A._pad_dense_input(B)
        res = A._mm(B_padded)
        return res[:, :col]
    else:
        B_t = B.t()
        if not isinstance(B_t, torch.sparse.SparseSemiStructuredTensor):
            raise AssertionError(
                f"expected SparseSemiStructuredTensor, got {type(B_t).__name__}"
            )
        row, col = A.shape
        A_padded = B._pad_dense_input(A)
        res = B_t._mm(A_padded.t()).t()
        return res[:row, :]


def semi_sparse_addmm(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) != 3:
        raise AssertionError(f"expected 3 args, got {len(args)}")
    bias, A, B = args
    if A.ndim != 2 or B.ndim != 2:
        raise NotImplementedError(
            "`SparseSemiStructuredTensor` matmul: Broadcasting is not implemented"
        )
    if bias.ndim != 1:
        raise NotImplementedError(
            f"`SparseSemiStructuredTensor` matmul: only bias dim=1 supported. Shape={bias.shape}"
        )
    if isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        raise NotImplementedError(
            "`SparseSemiStructuredTensor` matmul: only operand B of `addmm` can be sparse"
        )
    B_t = B.t()
    if not isinstance(B_t, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(B_t).__name__}"
        )
    row, _col = A.shape
    A_padded = B_t._pad_dense_input(A)
    result = B_t._mm(A_padded.t(), bias=bias).t()
    return result[:row, :]


def semi_sparse_linear(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) not in [2, 3]:
        raise AssertionError(f"expected 2 or 3 args, got {len(args)}")
    A, B = args[:2]
    bias = args[2] if len(args) == 3 else None

    shape = A.shape
    A_2d = A.view(-1, shape[-1])

    if bias is None:
        res = A_2d @ B.t()
    else:
        res = semi_sparse_addmm(
            func=None,
            types=None,
            args=[bias, A_2d, B.t()],
        )

    return res.view(*shape[:-1], -1)


def semi_sparse_scaled_mm(func, types, args=(), kwargs=None) -> torch.Tensor:
    # pull all args, excluding use_fast_accum flag if set.
    A, B, A_scale, B_scale, bias, scale_result, out_dtype = args[:7]

    if A.dtype != torch.float8_e4m3fn:
        raise AssertionError(f"expected A.dtype float8_e4m3fn, got {A.dtype}")
    if B.dtype != torch.float8_e4m3fn:
        raise AssertionError(f"expected B.dtype float8_e4m3fn, got {B.dtype}")
    # only cuSPARSELt supports float8_e4m3fn currently
    if not isinstance(A, torch.sparse.SparseSemiStructuredTensorCUSPARSELT):
        raise AssertionError(
            f"expected SparseSemiStructuredTensorCUSPARSELT, got {type(A).__name__}"
        )
    if A.packed is None:
        raise AssertionError("A.packed must not be None")
    # Currently we only support per-tensor scaling, with float32 scales
    if A_scale.numel() != 1 or B_scale.numel() != 1:
        raise AssertionError(
            f"expected A_scale and B_scale to have numel 1, got {A_scale.numel()} and {B_scale.numel()}"
        )
    if A_scale.dtype != torch.float32 or B_scale.dtype != torch.float32:
        raise AssertionError(
            f"expected A_scale and B_scale dtype float32, got {A_scale.dtype} and {B_scale.dtype}"
        )

    # cuSPARSELt lacks the A and B operand scaling support, so instead we use alpha to scale the result.
    # Note that this limits us to per-tensor scalig only.
    sparse_result = torch._cslt_sparse_mm(
        A.packed,
        B,
        alpha=A_scale * B_scale,
        out_dtype=out_dtype,
    )
    return sparse_result
