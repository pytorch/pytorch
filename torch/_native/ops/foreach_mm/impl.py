"""
Python override for aten::_foreach_mm that delegates to aten::_grouped_mm.

Reuses the grouped GEMM kernel (CUTLASS / cublasLt) without duplicating
C++ code. Works by stacking the TensorList inputs into 3D tensors and
calling _grouped_mm in 3D x 3D mode (no offsets).

The stacking is the main overhead vs a native C++ implementation.
We minimize it by:
  1. Checking if tensors are already contiguous views of a single
     allocation (e.g. from torch.split on a stacked tensor) -- if so,
     reconstruct the 3D view with no copy.
  2. Using torch.stack only as the fallback (one fused kernel, not G
     separate copies).
  3. Output splitting via unbind() is always zero-copy (views).
"""

import os

import torch

from ... import registry


def _foreach_mm_cond(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> bool:
    if os.environ.get("TORCH_FOREACH_MM_NATIVE_CPP") == "1":
        return False
    if len(self) < 2:
        return False

    first_a = self[0]
    first_b = mat2[0]

    if not first_a.is_cuda:
        return False
    if first_a.dim() != 2 or first_b.dim() != 2:
        return False

    # _grouped_mm 3D path only supports bf16 on SM90+
    if first_a.dtype != torch.bfloat16 or first_b.dtype != torch.bfloat16:
        return False

    props = torch.cuda.get_device_properties(first_a.device)
    if props.major < 9:
        return False

    return True


def _try_as_3d_view(tensors: list[torch.Tensor]) -> torch.Tensor | None:
    """Try to reconstruct a (G, rows, cols) view without copying.

    Succeeds when every tensor is a contiguous 2D slice of the same
    storage with uniform stride-0 step, i.e. they were produced by
    unbind/split/chunk on a contiguous 3D tensor.
    """
    if len(tensors) < 2:
        return None

    first = tensors[0]
    if not first.is_contiguous():
        return None

    storage = first.untyped_storage()
    rows, cols = first.shape
    elem_size = first.element_size()
    expected_stride0 = rows * cols * elem_size

    base_offset = first.storage_offset() * elem_size
    for i, t in enumerate(tensors):
        if t.untyped_storage().data_ptr() != storage.data_ptr():
            return None
        if t.shape != first.shape:
            return None
        if not t.is_contiguous():
            return None
        offset = t.storage_offset() * elem_size
        if offset != base_offset + i * expected_stride0:
            return None

    return torch.as_strided(
        first,
        (len(tensors), rows, cols),
        (rows * cols, cols, 1),
        storage_offset=first.storage_offset(),
    )


def _foreach_mm_impl_stack(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> list[torch.Tensor]:
    A = _try_as_3d_view(self)
    if A is None:
        A = torch.stack(self)

    B = _try_as_3d_view(mat2)
    if B is None:
        B = torch.stack(mat2)

    out_3d = torch._grouped_mm(A, B)
    return list(out_3d.unbind(0))


def _foreach_mm_impl_ptrs(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> list[torch.Tensor]:
    G = len(self)
    M, K = self[0].shape
    N = mat2[0].size(1)
    lda = self[0].stride(0)
    ldb = mat2[0].stride(0)

    a_ptrs = torch.tensor([t.data_ptr() for t in self], dtype=torch.int64)
    b_ptrs = torch.tensor([t.data_ptr() for t in mat2], dtype=torch.int64)

    # Pointer tensors are CPU; pass a dummy CUDA scalar to route dispatch to CUDA
    dummy = self[0].new_empty(0)
    return list(
        torch._foreach_mm_from_ptrs(
            dummy,
            a_ptrs,
            b_ptrs,
            M,
            N,
            K,
            G,
            lda,
            ldb,
        )
    )


def _foreach_mm_impl(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> list[torch.Tensor]:
    try:
        return _foreach_mm_impl_ptrs(self, mat2)
    except Exception:
        return _foreach_mm_impl_stack(self, mat2)


def register_to_dispatch() -> None:
    registry.register_op_override(
        "native",
        "aten",
        "_foreach_mm",
        "CUDA",
        cond=_foreach_mm_cond,
        impl=_foreach_mm_impl,
    )
