"""
Python Native DSL override for aten::_foreach_mm.

Delegates to aten::_grouped_mm by stacking the TensorList inputs into
3D tensors and calling _grouped_mm in 3D x 3D mode (no offsets).
Output is split back via unbind (zero-copy views).

A zero-copy fast path detects inputs that are already contiguous views
of a single storage (e.g. from unbind/split on a stacked tensor).

The main overhead vs the C++ _foreach_mm implementation is torch.stack,
which copies all matrix data into a contiguous allocation. For G=64 at
1024x1024 bf16 that's 134 MB per operand. The C++ path avoids this by
transferring only G*8 bytes of pointer values.
"""

import torch

from ... import registry


def _foreach_mm_cond(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> bool:
    if len(self) < 2:
        return False

    first_a = self[0]
    first_b = mat2[0]

    if not first_a.is_cuda:
        return False
    if first_a.dim() != 2 or first_b.dim() != 2:
        return False
    if first_a.dtype != torch.bfloat16 or first_b.dtype != torch.bfloat16:
        return False

    props = torch.cuda.get_device_properties(first_a.device)
    if props.major < 9:
        return False

    return True


def _try_as_3d_view(tensors: list[torch.Tensor]) -> torch.Tensor | None:
    """Try to reconstruct a (G, rows, cols) view without copying.

    Succeeds when every tensor is a contiguous 2D slice of the same
    storage with uniform stride-0 step.
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


def _foreach_mm_impl(
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


def register_to_dispatch() -> None:
    registry.register_op_override(
        "native",
        "aten",
        "_foreach_mm",
        "CUDA",
        cond=_foreach_mm_cond,
        impl=_foreach_mm_impl,
    )
