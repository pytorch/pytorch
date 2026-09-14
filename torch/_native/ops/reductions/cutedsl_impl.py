"""CuTeDSL override registrations for ``aten::sum`` / ``aten::prod`` (inner-tree).

CuTeDSL port of the Triton-style inner-tree reduction kernel that
otherwise lives in ``aten/src/ATen/native/cuda/ReduceSumProdKernel.cu``
(``try_inner_tree_reduction`` + the inner-tree kernels). This override
runs only when ``PYTORCH_SUM_INNER_TREE`` is set to a truthy value, which
is the feature-rollout gate for these operators -- it is *not* gated by the
global ``TORCH_DISABLE_NATIVE_JIT`` kill switch alone (that is handled by
``cutedsl_utils`` at registration time).

We register four ATen dispatcher entries on ``CUDA``:

* ``sum.dim_IntList`` / ``sum.IntList_out`` -- ``x.sum(dim=...)`` + ``.out``.
* ``prod.dim_int`` / ``prod.int_out`` -- ``x.prod(dim=...)`` + ``.out``.

sum/prod share the identical inner-tree geometry and eligibility; only the
combiner (``+`` vs ``*``) and its identity (``0`` vs ``1``) differ, threaded
through the kernel in ``inner_tree_kernel.py``.

``sum.dim_IntList`` is ``structured_delegate: sum.IntList_out``, so its
delegation to the ``.out`` kernel happens *below* the dispatcher; overriding
``.out`` alone would not intercept ``x.sum(dim=1)``. Both are separate
dispatcher entries and are registered explicitly (the same reason
``scatter_add_`` is registered separately from ``scatter_add.out``).

Eligibility mirrors ``try_inner_tree_reduction`` (ReduceSumProdKernel.cu):
build the reduction ``TensorIterator`` over ``(out, self)`` and accept only
the coalesced geometry the kernels know how to run -- a single contiguous
reduced (fastest) dimension whose non-reduced dims collapse to at most one
outer-strided dimension. Anything else (multi-dim reduction, non-contiguous
reduced dim, dtype-casting sum, integer/complex dtypes, non-collapsing outer
layout) falls through to aten.

Dispatch note: this CuTeDSL override is the only inner-tree path -- the ATen
CUDA backend has no inner-tree kernel of its own. When the rollout config is
disabled, no dispatcher override is installed. When enabled, accepted calls
run the CuTeDSL kernel and rejected calls fall through to unchanged ATen.
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
    """Return the reduction ``TensorIterator`` if ``(self, d, out)`` fits the
    kernel's expected geometry, else ``None``. ``d`` is already normalized
    and ``out`` is a keepdim-shaped output (broadcast-aligned with ``self``).

    Mirrors ``try_inner_tree_reduction``: build the reduction iterator, then
    require a single coalesced reduced (fastest) dimension with element
    stride 1 on the input, and at most one non-reduced (outer) dimension so
    the operands canonicalize to a single ``(M, N)`` view.
    """
    try:
        it = reduce_op(out, self)
    except RuntimeError:
        return None

    if it.numel == 0:
        return None
    if it.ndim == 0 or it.ndim > 2:
        return None

    input_index = it.ntensors - 1  # operands are (out, self); input is last
    es_in = it.element_strides(input_index)
    # Input must be contiguous on the reduced (fastest, dim-0) axis.
    if es_in[0] != 1:
        return None
    # Reduction must be on the fastest dim: either the whole tensor is the
    # reduction (ndim == 1) or the reduced-dim stride is the smallest.
    if it.ndim == 2 and not (es_in[0] < es_in[1]):
        return None
    return it


def _geometry(self: torch.Tensor, out: torch.Tensor, it: TensorIterator):
    """Recover ``(M, N, in_row_stride, out_row_stride)`` (element units) from
    the coalesced reduction iterator. ``M`` output rows, each a contiguous
    ``N``-element reduction; ``in_row_stride`` / ``out_row_stride`` step
    between rows of the canonical ``(M, N)`` / ``(M,)`` views."""
    m = out.numel()
    n = self.numel() // m if m else 0
    if it.ndim == 2:
        in_row_stride = it.element_strides(it.ntensors - 1)[1]
        out_row_stride = it.element_strides(0)[1]
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
    from .inner_tree_kernel import inner_tree_sum_into

    return inner_tree_sum_into


def _prod_into():
    from .inner_tree_kernel import inner_tree_prod_into

    return inner_tree_prod_into


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
