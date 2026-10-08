"""Register CUDA inner-tree overrides for ``aten::sum`` and ``aten::prod``.

``PYTORCH_SUM_INNER_TREE`` controls rollout; ``cutedsl_utils`` applies the global
native-JIT kill switch during registration. Direct and ``out=`` overloads are
registered separately because structured delegation occurs below the dispatcher.

Eligibility mirrors the former ``try_inner_tree_reduction``: TensorIterator must
coalesce the input to one contiguous reduced dimension and at most one outer
dimension. Unsupported geometry, dtype conversion, and integer or complex inputs
fall through to ATen.

Accepted calls canonicalize to ``(M, N)``. The ordered adapter uses the shared
fixed-DAG kernel for nonzero input row strides and unit-stride outputs, and the
legacy CuTeDSL reference for other accepted layouts.
"""

from __future__ import annotations

import os

import torch
from torch._subclasses.fake_tensor import is_fake_tensor
from torch._tensor_iterator import reduce_op, TensorIterator

from ... import cutedsl_utils as cu


# TODO: Move this rollout gate to a shared PyTorch config once eager native-op
# config ownership is established.
_INNER_TREE_ENABLED = os.getenv("PYTORCH_SUM_INNER_TREE", "") not in ("", "0")


# Reduction input dtypes the kernel handles. fp32/fp64 are the bitwise-
# validated set (the inner-tree bitwise tests upcast fp16/bf16/fp8 data to
# fp32 before summing, and fp64 sums in fp64). fp16/bf16 native reductions
# (acc in fp32) are functionally supported but not part of the bitwise
# contract.
_SUPPORTED_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)

# The kernel indexes rows/elements and sizes the launch grid with Int32. Decline
# reductions whose sizes would overflow that so we fall through to ATen rather
# than silently wrap (mirrors the magnitude guard from the CUDA review).
_INT32_MAX = 2**31 - 1


def _int32_indexing_safe(self: torch.Tensor, d: int) -> bool:
    """Decline giant reductions so the kernel's Int32 row/element indexing and
    launch grid never wrap. The two-kernel grid is ``m * num_batches``;
    ``num_batches`` is bounded above by ``ceil(n / (32 * vec_size))`` (the
    smallest possible per-batch width), so this bound covers every path."""
    n = self.shape[d]
    if n == 0:
        return True
    m = self.numel() // n
    vec_size = max(1, 16 // self.element_size())
    num_batches_ub = -(-n // (32 * vec_size))
    return n <= _INT32_MAX and m <= _INT32_MAX and m * num_batches_ub <= _INT32_MAX


def _normalize_dim(dim: int, ndim: int) -> int:
    return dim + ndim if dim < 0 else dim


def _single_dim(dim, ndim: int) -> int | None:
    """Return the one normalized reduced dim, or ``None`` if this is not a
    single-dim reduction (matches ``num_reduce_dims == 1``)."""
    if dim is None:
        return None
    if isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    if len(dims) != 1:
        return None
    return _normalize_dim(dims[0], ndim)


def _keepdim_out_shape(self: torch.Tensor, d: int) -> list[int]:
    shape = list(self.shape)
    shape[d] = 1
    return shape


def _out_keepdim_view(
    self: torch.Tensor, d: int, keepdim: bool, out: torch.Tensor
) -> torch.Tensor | None:
    """Return a keepdim-shaped ``(.., 1, ..)`` view of ``out`` so the kernel
    writes ``out``'s storage directly. ``unsqueeze`` is always a view (unlike
    ``reshape``, which may copy a non-contiguous ``out``). Returns ``None`` if
    ``out`` is not the expected reduced shape."""
    expected = _keepdim_out_shape(self, d)
    try:
        out_kd = out if keepdim else out.unsqueeze(d)
    except IndexError:
        return None
    return out_kd if list(out_kd.shape) == expected else None


def _eligibility(
    self: torch.Tensor, d: int, out: torch.Tensor
) -> TensorIterator | None:
    """Return the iterator if its operands canonicalize to ``(M, N)`` / ``(M,)``.

    ``d`` is normalized and ``out`` is keepdim-shaped. TensorIterator stores
    dimensions fastest-first and omits size-one dimensions.
    """
    try:
        it = reduce_op(out, self)
    except RuntimeError:
        return None

    if it.numel == 0:
        return None
    if it.ndim == 0 or it.ndim > 2:
        return None

    m = out.numel()
    n = self.numel() // m
    expected_shape = tuple(size for size in (n, m) if size != 1)
    if not expected_shape:
        expected_shape = (1,)
    if tuple(it.shape) != expected_shape:
        return None

    input_index = it.ntensors - 1  # operands are (out, self); input is last
    es_in = it.element_strides(input_index)
    # For N == 1, dim 0 is the row axis; retain the compact-row restriction.
    if es_in[0] != 1:
        return None
    if n > 1 and m > 1 and not (es_in[0] < es_in[1]):
        return None
    row_dim = 0 if n == 1 else 1
    if m > 1 and it.element_strides(0)[row_dim] == 0:
        return None
    return it


def _geometry(self: torch.Tensor, out: torch.Tensor, it: TensorIterator):
    """Recover ``(M, N, in_row_stride, out_row_stride)`` (element units) from
    the coalesced reduction iterator. ``M`` output rows, each a contiguous
    ``N``-element reduction; ``in_row_stride`` / ``out_row_stride`` step
    between rows of the canonical ``(M, N)`` / ``(M,)`` views."""
    m = out.numel()
    n = self.numel() // m if m else 0
    if m > 1:
        row_dim = 0 if n == 1 else 1
        in_row_stride = it.element_strides(it.ntensors - 1)[row_dim]
        out_row_stride = it.element_strides(0)[row_dim]
    else:
        in_row_stride = n
        out_row_stride = 1
    return m, n, in_row_stride, out_row_stride


def _base_cond_ok(self: torch.Tensor, dim, dtype) -> int | None:
    """Shared front gate. Returns the normalized reduced dim if the call is a
    candidate, else ``None``."""
    if not self.is_cuda or is_fake_tensor(self):
        return None
    # dtype-casting sum (an explicit out/result dtype) is out of scope.
    if dtype is not None:
        return None
    if self.dtype not in _SUPPORTED_DTYPES:
        return None
    d = _single_dim(dim, self.ndim)
    if d is None or not (0 <= d < self.ndim):
        return None
    if not _int32_indexing_safe(self, d):
        return None
    return d


def _cond(self, dim, keepdim=False, *, dtype=None) -> bool:
    d = _base_cond_ok(self, dim, dtype)
    if d is None:
        return False
    out = torch.empty(_keepdim_out_shape(self, d), dtype=self.dtype, device=self.device)
    return _eligibility(self, d, out) is not None


def _out_cond(self, dim, keepdim=False, *, dtype=None, out) -> bool:
    d = _base_cond_ok(self, dim, dtype)
    if d is None:
        return False
    if out.dtype != self.dtype or not out.is_cuda or is_fake_tensor(out):
        return False
    # The kernel writes ``out`` directly via a keepdim-aligned view; ``out``
    # must already have the reduced shape. Reject mis-shaped ``out`` and let
    # aten produce the proper error.
    out_kd = _out_keepdim_view(self, d, keepdim, out)
    if out_kd is None:
        return False
    return _eligibility(self, d, out_kd) is not None


def _sum_into():
    from .ordered import sum_into

    return sum_into


def _prod_into():
    from .ordered import prod_into

    return prod_into


def _run(self: torch.Tensor, d: int, out_kd: torch.Tensor, reduce_into) -> None:
    """Run ``reduce_into`` writing the reduction of ``self`` along ``d`` into
    the keepdim-shaped ``out_kd``. Caller has validated eligibility."""
    it = _eligibility(self, d, out_kd)
    if it is None:
        raise RuntimeError("cutedsl reduce: cond approved but iter rebuild failed")
    m, n, in_rs, out_rs = _geometry(self, out_kd, it)
    if m == 0 or n == 0:
        return
    in_2d = self.as_strided((m, n), (in_rs, 1))
    out_1d = out_kd.as_strided((m,), (out_rs,))
    reduce_into()(out_1d, in_2d)


def _make_impl(reduce_into):
    def _impl(self, dim, keepdim=False, *, dtype=None):
        d = _normalize_dim(dim[0] if not isinstance(dim, int) else dim, self.ndim)
        out_kd = torch.empty(
            _keepdim_out_shape(self, d), dtype=self.dtype, device=self.device
        )
        _run(self, d, out_kd, reduce_into)
        return out_kd if keepdim else out_kd.squeeze(d)

    return _impl


def _make_out_impl(reduce_into):
    def _out_impl(self, dim, keepdim=False, *, dtype=None, out):
        d = _normalize_dim(dim[0] if not isinstance(dim, int) else dim, self.ndim)
        out_kd = _out_keepdim_view(self, d, keepdim, out)
        if out_kd is None:
            # _out_cond guarantees a reduced-shape out; raise explicitly rather
            # than assert so the invariant survives `python -O`.
            raise AssertionError("_out_cond guaranteed a reduced-shape out")
        _run(self, d, out_kd, reduce_into)
        return out

    return _out_impl


def register_to_dispatch() -> None:
    if not _INNER_TREE_ENABLED:
        return

    # Eligibility (_cond/_out_cond) is reduction-agnostic; only the kernel
    # entry differs between sum and prod.
    cu.register_op_override(
        "aten", "sum.dim_IntList", "CUDA", cond=_cond, impl=_make_impl(_sum_into)
    )
    cu.register_op_override(
        "aten",
        "sum.IntList_out",
        "CUDA",
        cond=_out_cond,
        impl=_make_out_impl(_sum_into),
    )
    cu.register_op_override(
        "aten", "prod.dim_int", "CUDA", cond=_cond, impl=_make_impl(_prod_into)
    )
    cu.register_op_override(
        "aten", "prod.int_out", "CUDA", cond=_out_cond, impl=_make_out_impl(_prod_into)
    )
