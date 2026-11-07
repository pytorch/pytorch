"""FlyDSL override registrations for row-shaped ``scatter_add`` / ``index_add``.

Both ops reduce to ``dst[index[i], :] += src[i, :]`` once the operands
coalesce to a 2D (row, contiguous slice) layout. The layout analysis is the
one the CuTeDSL sibling uses, and mirrors ATen's
``fast_scatter_kernel_eligible``. ``index_add(dim=0, alpha=1)`` is analyzed as
``scatter_add`` with the 1D index expanded over ``source``, just like ATen's
CUDA >= 12.8 delegation.

The cond accepts only the narrow v1 contract; everything else falls through to
ATen, which also raises ATen's own errors:

* ROCm on ``_SUPPORTED_ARCHES``, all operands on one device, no COW operands,
  non-deterministic mode only;
* fp16/bf16/fp32 with ``src.dtype == dst.dtype``; int32/int64 index;
* ATen's original shape checks, no internal overlap in the destination, and no
  overlap between the destination and ``src``/``index`` (``out`` must be
  ``self`` or disjoint from it);
* coalesced slice of ``N`` elements with ``N * itemsize >= 256`` bytes, a
  multiple of 16 bytes, and at most ``MAX_STEPS_PER_LANE`` 4-byte steps per
  lane (``flydsl_geometry.py``, shared with the kernel);
* ``src`` base and row stride 16-byte aligned; destination base and row
  stride 4-byte aligned. The kernel needs only 4-byte alignment and whole
  words; the stricter row and ``src`` rules are kept from v1 and could be
  relaxed with tests;
* the byte span of every operand fits a signed int32 (buffer addressing);
* non-empty work, at least ``_MIN_SRC_BYTES`` of source for the dtype (no
  fp32), so that the call is large enough to win through the native route.

``index_add`` additionally declines exactly the calls ATen routes to its
sorted FP32 reduction on gfx942/gfx950 (``index_add_sorted_rocm`` in
``Indexing.cu``), which is faster for the duplicate-heavy inputs it targets.

Out-of-range indices abort in the kernel, matching ATen's ROCm
``CUDA_KERNEL_ASSERT`` (``abort()``); see ``flydsl_kernels.py``.

Further guards:

* the overrides register only for validated FlyDSL releases
  (``_VALIDATED_FLYDSL_RELEASES``); the shared gate accepts any stable 0.3.x,
  but releases before 0.3.3 lack ``fx.atomic_add``;
* FlyDSL compiles for one process-wide target, so calls whose destination
  device has a different arch decline;
* functional variants allocate a contiguous result (as ATen does), so they
  require a contiguous ``self``, whose approved layout the result then shares.

All conds read ``data_ptr()``/COW state, so they are eager-only: under
tracing they raise and the registry treats that as "no match".
"""

from __future__ import annotations

import threading
import weakref
from typing import NamedTuple

import torch

from ... import flydsl_utils as fu
from ...common_utils import check_native_version_skip
from .cutedsl_impl import _MIN_SLICE_BYTES, _normalize_dim, _scatter_add_eligibility
from .flydsl_geometry import MAX_STEPS_PER_LANE, steps_per_lane


_SUPPORTED_ARCHES = ("gfx942", "gfx950")
# FlyDSL releases (major, minor, patch) this override was written and checked
# against; patch releases of these (e.g. 0.3.4.1) are included.
_VALIDATED_FLYDSL_RELEASES = ((0, 3, 4),)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SRC_ALIGN_BYTES = 16
_DST_ALIGN_BYTES = 4
_ROW_GRANULE_BYTES = 16
_INT32_MAX = (1 << 31) - 1

# Minimum source bytes (M_src * N * itemsize) per dtype for the native route to
# dispatch; None never dispatches. The kernel has no such limit: this is where
# its gain covers the native route's host cost (about 25 us per call against
# 4 us for ATen) in the worst index distribution, which metadata cannot see.
# MI355X (gfx950), native index_add_ vs ATen eager, per-call timing:
# * bf16/fp16: 1.1-1.9x from 64 MiB with uniform or MoE indices, but with
#   half or all rows on one destination row 0.84x at 64 MiB, 0.96x at
#   256 MiB and 0.99x at 512 MiB (about run-to-run noise). 256 MiB keeps
#   most of the uniform/MoE wins (1.6-1.9x there) at that 0.96x worst case;
# * fp32: at most 1.12x, and 0.97-0.99x with all rows on one destination
#   even at 512 MiB-1 GiB, so it is not dispatched.
_MIN_SRC_BYTES: dict[torch.dtype, int | None] = {
    torch.bfloat16: 256 << 20,
    torch.float16: 256 << 20,
    torch.float32: None,
}

# Mirrors of the Indexing.cu constants that gate index_add_sorted_rocm.
_ATEN_SORTED_INDEX_ADD_ARCHES = ("gfx942", "gfx950")
_ATEN_SORTED_INDEX_ADD_MIN_INDICES = 1_000_000
_ATEN_SORTED_INDEX_ADD_INNER_SIZES = (128, 256)


def _flydsl_release_validated() -> bool:
    if check_native_version_skip():
        return True
    version = fu.runtime_version()
    return (
        version is not None
        and not version.is_prerelease
        and tuple(version.release[:3]) in _VALIDATED_FLYDSL_RELEASES
    )


def _compile_target_matches(device_index: int) -> bool:
    """FlyDSL compiles for one process-wide target; it must be this device's."""
    from .flydsl_kernels import compile_target_arch

    device_arch = fu._get_flydsl_device_arch(device_index)
    return device_arch is not None and device_arch == compile_target_arch()


def _wave_size(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).warp_size


def _has_python_dispatch(t: torch.Tensor) -> bool:
    return torch._C._dispatch_keys(t).has(torch._C.DispatchKey.Python)


def _common_ok(*tensors: torch.Tensor) -> bool:
    """Environment and device checks shared by every override."""
    if torch.version.hip is None:
        return False
    if torch.are_deterministic_algorithms_enabled():
        return False
    first = tensors[0]
    if not first.is_cuda:
        return False
    if any(t.device != first.device for t in tensors):
        return False
    # _is_cow_tensor raises for tensors with the Python dispatch key (tensor
    # subclasses with __torch_dispatch__, FakeTensors), and the eager router
    # propagates cond exceptions. Decline them so ATen handles the call.
    if any(_has_python_dispatch(t) for t in tensors):
        return False
    if any(
        torch._C._is_cow_tensor(t)  # pyrefly: ignore[missing-attribute]
        for t in tensors
    ):
        return False
    device_index = first.device.index
    if device_index is None:
        return False
    return fu._is_supported_arch(device_index, _SUPPORTED_ARCHES)


def _alias_ok(dst: torch.Tensor, plan: _RowPlan, *inputs: torch.Tensor) -> bool:
    """The kernel's writes are disjoint, and ``dst`` shares no storage with inputs.

    ``_debug_has_internal_overlap`` returns 0 (no), 1 (yes) or 2 (too hard);
    2 covers every non-dense view, such as padded rows. The kernel writes only
    the plan's rows: ``N`` unit-stride elements ``sd`` apart, which are
    disjoint when ``sd >= N``. ``torch._C._overlaps`` is conservative: any
    shared storage counts.
    """
    if torch._debug_has_internal_overlap(dst) == 1:
        return False
    if plan.m_dst > 1 and plan.dst_row_stride < plan.n:
        return False
    return not any(torch._C._overlaps(dst, t) for t in inputs)


def _out_alias_ok(self: torch.Tensor, out: torch.Tensor) -> bool:
    """``out`` must be ``self`` itself or fully disjoint from it."""
    return out.is_set_to(self) or not torch._C._overlaps(out, self)


def _scatter_shape_ok(
    self: torch.Tensor, d: int, index: torch.Tensor, src: torch.Tensor
) -> bool:
    """ATen's scatter_shape_check for a non-empty index (ScatterGatherChecks.h)."""
    if not (self.ndim == index.ndim == src.ndim) or self.ndim == 0:
        return False
    for i in range(self.ndim):
        if index.size(i) > src.size(i):
            return False
        if i != d and index.size(i) > self.size(i):
            return False
    return True


def _span_elems(rows: int, row_stride: int, n: int) -> int:
    return (rows - 1) * row_stride + n


class _RowPlan(NamedTuple):
    """An approved row layout: ``dst`` rows of ``n`` unit-stride elements
    ``dst_row_stride`` apart, read from ``src`` rows ``src_row_stride`` apart."""

    m_dst: int
    n: int
    dst_row_stride: int
    m_src: int
    src_row_stride: int

    def flat_dst(self, dst: torch.Tensor) -> torch.Tensor:
        """1D view over the ``dst`` elements the kernel addresses."""
        span = _span_elems(self.m_dst, self.dst_row_stride, self.n)
        return dst.as_strided((span,), (1,))

    def flat_src(self, src: torch.Tensor) -> torch.Tensor:
        span = _span_elems(self.m_src, self.src_row_stride, self.n)
        return src.as_strided((span,), (1,))


def _plan_rows(
    dst: torch.Tensor,
    src: torch.Tensor,
    m_dst: int,
    m_src: int,
    n: int,
    sd: int,
    ss: int,
) -> _RowPlan | None:
    """The kernel's size, alignment and span limits for a row layout. Every
    layout route (general analysis and the ``index_add`` row fast path) comes
    through here, so the row-width contract is enforced in one place."""
    elem = dst.element_size()
    if n * elem < _MIN_SLICE_BYTES or (n * elem) % _ROW_GRANULE_BYTES:
        return None
    wave = _wave_size(dst.device.index)
    if steps_per_lane(n, elem, wave) > MAX_STEPS_PER_LANE:
        return None
    if src.data_ptr() % _SRC_ALIGN_BYTES or (ss * elem) % _SRC_ALIGN_BYTES:
        return None
    if dst.data_ptr() % _DST_ALIGN_BYTES or (sd * elem) % _DST_ALIGN_BYTES:
        return None
    if m_dst == 0 or m_src == 0 or m_dst > _INT32_MAX or m_src > _INT32_MAX:
        return None
    if _span_elems(m_dst, sd, n) * elem > _INT32_MAX:
        return None
    if _span_elems(m_src, ss, n) * elem > _INT32_MAX:
        return None
    return _RowPlan(m_dst, n, sd, m_src, ss)


def _geometry(
    dst: torch.Tensor, d: int, index_for_layout: torch.Tensor, src: torch.Tensor
) -> _RowPlan | None:
    """General layout analysis (TensorIterator coalescing, as ATen does)."""
    it = _scatter_add_eligibility(dst, d, index_for_layout, src)
    if it is None or it.numel == 0:
        return None
    n, m_src = it.shape[0], it.shape[1]
    return _plan_rows(dst, src, dst.shape[d], m_src, n, dst.stride(d), src.stride(d))


def _row_layout(t: torch.Tensor) -> tuple[int, int, int] | None:
    """``(rows, n, row_stride)`` if each dim-0 slice of ``t`` is one
    unit-stride run of ``n`` elements in row-major order, else None."""
    shape, strides = t.shape, t.stride()
    n = 1
    for i in range(len(shape) - 1, 0, -1):
        if shape[i] != 1 and strides[i] != n:
            return None
        n *= shape[i]
    return shape[0], n, strides[0]


def _index_add_geometry(
    dst: torch.Tensor, index: torch.Tensor, source: torch.Tensor
) -> _RowPlan | None:
    """``index_add(dim=0)`` layout. Row layouts, the common case, are read off
    the strides directly; anything else goes through the general analysis."""
    dst_rows = _row_layout(dst)
    src_rows = _row_layout(source)
    if dst_rows is not None and src_rows is not None:
        # Same inner shape (checked by the caller), so the same n and order.
        return _plan_rows(
            dst,
            source,
            dst_rows[0],
            src_rows[0],
            dst_rows[1],
            dst_rows[2],
            src_rows[2],
        )
    return _geometry(dst, 0, _expanded_index(index, source), source)


def _index_1d_ok(m_src: int, index: torch.Tensor) -> bool:
    """The kernel reads ``M_src`` consecutive index entries."""
    return m_src * index.element_size() <= _INT32_MAX


def _worth_dispatching(dtype: torch.dtype, plan: _RowPlan) -> bool:
    """Whether the call is large enough for the kernel's gain to cover the
    native route's host cost (see ``_MIN_SRC_BYTES``)."""
    floor = _MIN_SRC_BYTES.get(dtype)
    return floor is not None and plan.m_src * plan.n * dtype.itemsize >= floor


def _may_be_worth_dispatching(dtype: torch.dtype, numel: int) -> bool:
    """Early, cheap form of ``_worth_dispatching`` on the call's element count
    (an upper bound on the plan's), so most calls the floor will reject skip
    the layout analysis: declined calls pay the router for nothing."""
    floor = _MIN_SRC_BYTES.get(dtype)
    return floor is not None and numel * dtype.itemsize >= floor


# The router runs a cond and then, on a match, its impl with the same
# arguments. The cond's layout analysis is the costly part of a dispatch, so
# the cond leaves the approved plan here and the impl takes it instead of
# recomputing it. The entry holds weak references, is consumed (or replaced)
# by the next take/approval on this thread, and is used only when the impl
# sees the same objects at the same addresses with the same shapes and
# strides; otherwise the impl recomputes the plan.
_approved = threading.local()


def _layout_key(*tensors: torch.Tensor) -> tuple:
    return tuple((t.data_ptr(), t.shape, t.stride()) for t in tensors)


def _approve(
    kind: str,
    key: torch.Tensor,
    index: torch.Tensor,
    src: torch.Tensor,
    plan: _RowPlan | None,
) -> bool:
    """Record ``plan`` for the ``kind`` impl that writes through ``key``."""
    if plan is None:
        return False
    _approved.entry = (
        kind,
        weakref.ref(key),
        weakref.ref(index),
        weakref.ref(src),
        _layout_key(key, index, src),
        plan,
    )
    return True


def _take_approved(
    kind: str, key: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> _RowPlan | None:
    entry = getattr(_approved, "entry", None)
    _approved.entry = None
    if entry is None:
        return None
    entry_kind, key_ref, index_ref, src_ref, layout, plan = entry
    if entry_kind != kind:
        return None
    if key_ref() is not key or index_ref() is not index or src_ref() is not src:
        return None
    if layout != _layout_key(key, index, src):
        return None
    return plan


def _launch(
    dst: torch.Tensor, index_1d: torch.Tensor, src: torch.Tensor, plan: _RowPlan
) -> None:
    # The cond checked the compile target for this device.
    from .flydsl_kernels import launch_rows

    launch_rows(
        plan.flat_dst(dst),
        index_1d,
        plan.flat_src(src),
        plan.m_src,
        plan.m_dst,
        plan.n,
        plan.dst_row_stride,
        plan.src_row_stride,
    )


# ---------------------------------------------------------------------------
# scatter_add
# ---------------------------------------------------------------------------


def _scatter_index_1d(index: torch.Tensor, m_src: int) -> torch.Tensor:
    # Eligibility proved index has stride 1 along d and stride 0 elsewhere
    # after coalescing, so its M_src entries are consecutive from offset 0.
    return index.as_strided((m_src,), (1,))


def _scatter_add_ok(
    self: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> _RowPlan | None:
    """The row plan for ``self`` if the call fits and is worth it, else None."""
    if not isinstance(src, torch.Tensor) or index.numel() == 0:
        return None
    if self.dtype not in _SUPPORTED_DTYPES or src.dtype != self.dtype:
        return None
    # The kernel's work is the index's elements (one source read each).
    if not _may_be_worth_dispatching(self.dtype, index.numel()):
        return None
    if index.dtype not in (torch.int32, torch.int64):
        return None
    if not _common_ok(self, index, src):
        return None
    if self.ndim == 0 or not -self.ndim <= dim < self.ndim:
        return None
    d = _normalize_dim(dim, self.ndim)
    if not _scatter_shape_ok(self, d, index, src):
        return None
    plan = _geometry(self, d, index, src)
    if plan is None or not _worth_dispatching(self.dtype, plan):
        return None
    if not _index_1d_ok(plan.m_src, index):
        return None
    if not _alias_ok(self, plan, index, src):
        return None
    # Last: the first call imports FlyDSL.
    if not _compile_target_matches(self.device.index):
        return None
    return plan


def _scatter_add_cond(self, dim, index, src, *args, **kwargs) -> bool:
    # The functional result is a contiguous clone of self; requiring a
    # contiguous self makes the approved layout the result's layout.
    return self.is_contiguous() and _approve(
        "scatter", self, index, src, _scatter_add_ok(self, int(dim), index, src)
    )


def _scatter_add_inplace_cond(self, dim, index, src, *args, **kwargs) -> bool:
    plan = _scatter_add_ok(self, int(dim), index, src)
    return _approve("scatter", self, index, src, plan)


def _scatter_add_out_cond(self, dim, index, src, *, out) -> bool:
    if out.dtype != self.dtype or out.shape != self.shape:
        return False
    if _scatter_add_ok(self, int(dim), index, src) is None:
        return False
    if not _common_ok(out, self) or not _out_alias_ok(self, out):
        return False
    d = _normalize_dim(int(dim), out.ndim)
    plan = _geometry(out, d, index, src)
    if plan is None or not _alias_ok(out, plan, index, src):
        return False
    return _approve("scatter", out, index, src, plan)


def _run_scatter_add(
    dst: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    plan: _RowPlan | None,
) -> torch.Tensor:
    """Apply the kernel to ``dst``; ``plan`` is the cond's, if it was handed over."""
    if plan is None:
        plan = _geometry(dst, _normalize_dim(dim, dst.ndim), index, src)
        if plan is None:
            raise RuntimeError(
                "scatter_add flydsl: cond approved but layout check failed"
            )
    _launch(dst, _scatter_index_1d(index, plan.m_src), src, plan)
    return dst


def _scatter_add_impl(self, dim, index, src, *args, **kwargs):
    # Approved on self, which is contiguous; the contiguous clone has its layout.
    plan = _take_approved("scatter", self, index, src)
    out = self.clone(memory_format=torch.contiguous_format)
    return _run_scatter_add(out, int(dim), index, src, plan)


def _scatter_add_out_impl(self, dim, index, src, *, out):
    plan = _take_approved("scatter", out, index, src)
    if not out.is_set_to(self):
        out.copy_(self)
    return _run_scatter_add(out, int(dim), index, src, plan)


def _scatter_add_inplace_impl(self, dim, index, src, *args, **kwargs):
    plan = _take_approved("scatter", self, index, src)
    return _run_scatter_add(self, int(dim), index, src, plan)


# ---------------------------------------------------------------------------
# index_add
# ---------------------------------------------------------------------------


def _alpha_is_one(alpha) -> bool:
    # bool is excluded on purpose: ATen treats it as its own Scalar kind.
    return type(alpha) in (int, float) and alpha == 1


def _aten_sorted_index_add_would_run(
    result_contiguous: bool,
    self: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
) -> bool:
    """Mirror of ``index_add_sorted_rocm``'s acceptance test (dim 0, alpha 1).

    Fast reject first; the arch query is last because it is the only one that
    touches device properties.
    """
    num_indices = index.numel()
    if num_indices < _ATEN_SORTED_INDEX_ADD_MIN_INDICES:
        return False
    if num_indices > _INT32_MAX:
        return False
    if not (result_contiguous and source.is_contiguous() and index.is_contiguous()):
        return False
    inner = 1
    for s in self.shape[1:]:
        inner *= s
    if inner not in _ATEN_SORTED_INDEX_ADD_INNER_SIZES:
        return False
    if not 0 < self.size(0) <= _INT32_MAX:
        return False
    return fu._is_supported_arch(self.device.index, _ATEN_SORTED_INDEX_ADD_ARCHES)


def _expanded_index(index: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    shape = [1] * source.ndim
    shape[0] = index.numel()
    return index.view(shape).expand_as(source)


def _index_add_ok(
    self: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    source: torch.Tensor,
    alpha,
    *,
    result_contiguous: bool,
) -> _RowPlan | None:
    """The row plan for ``self`` if the call fits and is worth it, else None."""
    if not _alpha_is_one(alpha):
        return None
    if index.ndim != 1 or index.numel() == 0 or not index.is_contiguous():
        return None
    if self.dtype not in _SUPPORTED_DTYPES or source.dtype != self.dtype:
        return None
    if not _may_be_worth_dispatching(self.dtype, source.numel()):
        return None
    if index.dtype not in (torch.int32, torch.int64):
        return None
    if self.ndim == 0 or source.ndim != self.ndim:
        return None
    if not -self.ndim <= dim < self.ndim or _normalize_dim(dim, self.ndim) != 0:
        return None
    # ATen's index_func_meta_impl shape rules, for dim 0.
    if source.size(0) != index.numel() or self.shape[1:] != source.shape[1:]:
        return None
    if not _common_ok(self, index, source):
        return None
    if _aten_sorted_index_add_would_run(result_contiguous, self, index, source):
        return None
    plan = _index_add_geometry(self, index, source)
    if plan is None or not _worth_dispatching(self.dtype, plan):
        return None
    if not _index_1d_ok(index.numel(), index):
        return None
    if not _alias_ok(self, plan, index, source):
        return None
    # Last: the first call imports FlyDSL.
    if not _compile_target_matches(self.device.index):
        return None
    return plan


def _index_add_cond(self, dim, index, source, *, alpha=1) -> bool:
    # See _scatter_add_cond: the functional result is a contiguous clone.
    return self.is_contiguous() and _approve(
        "index",
        self,
        index,
        source,
        _index_add_ok(self, int(dim), index, source, alpha, result_contiguous=True),
    )


def _index_add_inplace_cond(self, dim, index, source, *, alpha=1) -> bool:
    plan = _index_add_ok(
        self,
        int(dim),
        index,
        source,
        alpha,
        result_contiguous=self.is_contiguous(),
    )
    return _approve("index", self, index, source, plan)


def _index_add_out_cond(self, dim, index, source, *, alpha=1, out) -> bool:
    if out.dtype != self.dtype or out.shape != self.shape:
        return False
    if (
        _index_add_ok(
            self,
            int(dim),
            index,
            source,
            alpha,
            result_contiguous=out.is_contiguous(),
        )
        is None
    ):
        return False
    if not _common_ok(out, self) or not _out_alias_ok(self, out):
        return False
    plan = _index_add_geometry(out, index, source)
    if plan is None or not _alias_ok(out, plan, index, source):
        return False
    return _approve("index", out, index, source, plan)


def _run_index_add(
    dst: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
    plan: _RowPlan | None,
) -> torch.Tensor:
    """Apply the kernel to ``dst``; ``plan`` is the cond's, if it was handed over."""
    if plan is None:
        plan = _index_add_geometry(dst, index, source)
        if plan is None:
            raise RuntimeError(
                "index_add flydsl: cond approved but layout check failed"
            )
    _launch(dst, index, source, plan)
    return dst


def _index_add_impl(self, dim, index, source, *, alpha=1):
    # Approved on self, which is contiguous; the contiguous clone has its layout.
    plan = _take_approved("index", self, index, source)
    out = self.clone(memory_format=torch.contiguous_format)
    return _run_index_add(out, index, source, plan)


def _index_add_out_impl(self, dim, index, source, *, alpha=1, out):
    plan = _take_approved("index", out, index, source)
    if not out.is_set_to(self):
        out.copy_(self)
    return _run_index_add(out, index, source, plan)


def _index_add_inplace_impl(self, dim, index, source, *, alpha=1):
    plan = _take_approved("index", self, index, source)
    return _run_index_add(self, index, source, plan)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


_OVERRIDES = (
    # The functional/.out/in-place entries are separate dispatcher ops; the
    # structured in-place delegation to .out happens below the dispatcher.
    ("scatter_add", _scatter_add_cond, _scatter_add_impl),
    ("scatter_add.out", _scatter_add_out_cond, _scatter_add_out_impl),
    ("scatter_add_", _scatter_add_inplace_cond, _scatter_add_inplace_impl),
    ("index_add", _index_add_cond, _index_add_impl),
    ("index_add.out", _index_add_out_cond, _index_add_out_impl),
    ("index_add_", _index_add_inplace_cond, _index_add_inplace_impl),
)


def register_to_dispatch() -> None:
    # Checked here, not in flydsl_utils: other FlyDSL ops may support releases
    # this one does not. Reads package metadata only; FlyDSL is not imported.
    if not _flydsl_release_validated():
        return
    for op_symbol, cond, impl in _OVERRIDES:
        fu.register_op_override(
            "aten",
            op_symbol,
            "CUDA",
            cond=cond,
            impl=impl,
            allow_multiple_override=True,
        )
