import contextlib

import torch

__all__ = [
    "fallback_dispatcher",
    "sparse24_values",
    "sparse24_indices",
    "sparse24_t",
    "sparse24_view",
    "sparse24_detach",
    "sparse24_mm",
    "sparse24_addmm",
    "sparse24_linear",
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


def sparse24_values(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 1
    A = args[0]
    assert isinstance(A, torch.sparse.SparseSemiStructuredTensor)
    assert A.packed is not None
    if A.meta is None:
        m, k = A.shape
        num_kept_elements = m * k // 2
        return A.packed[:num_kept_elements:].view(m, -1)
    else:
        return A.packed.detach()


def sparse24_indices(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 1
    A = args[0]
    assert isinstance(A, torch.sparse.SparseSemiStructuredTensor)
    assert A.packed is not None
    if A.meta is None:
        m, k = A.shape
        num_kept_elements = m * k // 2
        metadata = A.packed[num_kept_elements:].view(m, -1)
        return metadata.view(torch.int32 if A.dtype == torch.int32 else torch.int16)
    else:
        return A.meta


def sparse24_t(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 1
    self = args[0]
    assert isinstance(self, torch.sparse.SparseSemiStructuredTensor)
    assert len(self.shape) == 2
    # Because we cannot go from the compressed representation back to the dense representation currently,
    # we just keep track of how many times we have been transposed. Depending on whether the sparse matrix
    # is the first or second argument, we expect an even / odd number of calls to transpose respectively.
    return self.__class__(
        torch.Size([self.shape[-1], self.shape[0]]),
        packed=self.packed_t,
        meta=self.meta_t,
        packed_t=self.packed,
        meta_t=self.meta,
        threads_masks=self.threads_masks.transpose(0, 1)
        if self.threads_masks is not None
        else None,
        fuse_transpose_cusparselt=args[0].fuse_transpose_cusparselt,
        alg_id_cusparselt=args[0].alg_id_cusparselt,
    )


def sparse24_view(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 2
    self, shape = args
    if tuple(shape) != self.shape:
        raise NotImplementedError(
            f"`view` is not implemented for SparseSemiStructuredTensor, except for the dummy case (shape={shape})"
        )
    return self


def sparse24_detach(func, types, args, kwargs) -> torch.Tensor:
    assert len(args) == 1
    self = args[0]
    return self.__class__(
        shape=self.shape,
        packed=self.packed,
        meta=self.meta,
        packed_t=self.packed_t,
        meta_t=self.meta_t,
        threads_masks=self.threads_masks,
        requires_grad=False,
    )


def sparse24_mm(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 2
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
        assert isinstance(B_t, torch.sparse.SparseSemiStructuredTensor)
        row, col = A.shape
        A_padded = B._pad_dense_input(A)
        res = B_t._mm(A_padded.t()).t()
        return res[:row, :]


def sparse24_addmm(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 3
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
    assert isinstance(B_t, torch.sparse.SparseSemiStructuredTensor)
    row, col = A.shape
    A_padded = B_t._pad_dense_input(A)
    result = B_t._mm(A_padded.t(), bias=bias).t()
    return result[:row, :]


def sparse24_linear(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) in [2, 3]
    A, B = args[:2]
    bias = args[2] if len(args) == 3 else None

    shape = A.shape
    A_2d = A.view(-1, shape[-1])

    if bias is None:
        res = A_2d @ B.t()
    else:
        res = sparse24_addmm(
            func=None,
            types=None,
            args=[bias, A_2d, B.t()],
        )

    return res.view(*shape[:-1], -1)
