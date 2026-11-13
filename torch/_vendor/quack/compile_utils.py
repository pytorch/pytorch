# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

from typing import Optional

import cutlass.cute as cute


def make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1) -> Optional[cute.Tensor]:
    """Build a fake CuTe tensor with dynamic (sym) strides for tensor-free compilation.

    ``leading_dim`` selects the dim whose stride is statically 1 (matching
    ``from_dlpack(...).mark_layout_dynamic(leading_dim=...)``). Pass
    ``leading_dim=None`` for a fully-dynamic layout with no static stride-1 dim
    (matching ``mark_layout_dynamic()`` on a tensor without a contiguous dim).

    ``divisibility`` is in elements; ``assumed_align`` (bytes) is
    ``divisibility * dtype.width // 8``, floored to at least 1 so sub-byte
    dtypes (bool width=1, int4/fp4 width=4) never claim more alignment than
    ``divisibility`` elements guarantee. Callers pick ``divisibility`` for wide
    dtypes' vectorization, so bool tensors flowing through generic paths only
    get the always-safe 1-byte claim; floor (not ceil) keeps non-divisible
    sub-byte cases from over-claiming.
    """
    if dtype is None:
        return None
    if leading_dim is not None and leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    assumed_align = max(divisibility * dtype.width // 8, 1)
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=assumed_align)


def div_for_dtype(dtype):
    """16-byte alignment: divisibility in elements = 128 // dtype_width_bits."""
    return 128 // dtype.width


def fake_batched(dtype, x, y, l, leading_dim, divisibility):  # noqa: E741
    """Batch-first (l, x, y) fake tensor; ``leading_dim`` indexes into (x, y).

    Batched tensors cross the FFI boundary in the caller's natural torch order
    (l, x, y) and the kernel rotates them to (x, y, l) at trace time
    (GemmBase.rotate_batch_last), so the batch dim always prepends — hence the
    ``+ 1``. Pass ``l=None`` for a varlen-flattened 2D (x, y) tensor.
    """
    if l is None:
        return make_fake_tensor(dtype, (x, y), leading_dim=leading_dim, divisibility=divisibility)
    return make_fake_tensor(
        dtype, (l, x, y), leading_dim=leading_dim + 1, divisibility=divisibility
    )


def make_fake_stream():
    """Fake CUDA stream for tensor-free compilation (real stream comes from the TVM FFI env)."""
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
