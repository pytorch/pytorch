"""CuTeDSL override registrations for ``aten::scatter_add``.

Two kernels for the fast-path layout (self/src 2D-shaped after
coalescing, with self contiguous on the slice axis and src either
contiguous or row-strided), tried in order:

1. **TMA** (``tma_kernel.py``, sm_90+): uses ``cp.reduce.async.bulk`` to
   offload the whole reduction to the TMA unit, 3-7x faster than aten on
   high-contention embedding-bag-grad-style workloads.
2. **vec-scatter** (``vec_scatter_kernel.py``, any SM): warp-per-row
   scheduling with ``atomicAdd`` (scalar fp32, paired x2 halves). Covers
   pre-sm_90, and sm_90+ shapes where the TMA alignment constraint isn't
   met.

Eligibility mirrors aten's ``fast_scatter_add_kernel_eligible``
(IndexKernelUtils.h): we restride ``self`` so its scatter-axis stride
is 0 and shape matches ``index``, then build a ``TensorIterator`` on
``(self_restrided, src_restrided, index)``. TensorIterator's coalescing
and stride-magnitude reordering produces a 2D iterator iff the layout
is kernel-friendly; we then pattern-match the resulting iter strides
to confirm slice/index-axis assignment. This delegates the layout
analysis to ``TensorIterator`` rather than reimplementing it.

Anything else (non-2D-coalescing layout, dtype mismatch, deterministic
mode) falls through to aten. Aten is competitive or better on those
shapes, so we intentionally don't override them.

In-place (``scatter_add_``) is registered explicitly; PyTorch's
structured-delegate from ``scatter_add_`` to ``scatter_add.out`` happens
below the dispatcher, so overriding the ``.out`` variant alone doesn't
intercept the in-place method.
"""

import functools
import importlib.util

import torch
from torch._tensor_iterator import TensorIterator, TensorIteratorConfig

from ... import cutedsl_utils as cu
from ...registry import _OpCondFn, _OpImplFn


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@functools.cache
def _has_cutedsl() -> bool:
    try:
        return (
            importlib.util.find_spec("cutlass") is not None
            and importlib.util.find_spec("tvm_ffi") is not None
        )
    except ModuleNotFoundError:
        return False


@functools.cache
def _has_sm90_plus() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9


def _deterministic() -> bool:
    # Our kernels use non-deterministic atomicAdd. Under
    # torch.use_deterministic_algorithms(True), aten's scatter_add falls
    # into _scatter_via_index_put instead; we must decline to preserve
    # that behavior.
    return torch.are_deterministic_algorithms_enabled()


def _any_cow(*tensors: torch.Tensor) -> bool:
    return any(
        torch._C._is_cow_tensor(t)  # pyrefly: ignore[missing-attribute]
        for t in tensors
    )


def _base_cond_ok(*tensors: torch.Tensor) -> bool:
    """Pre-checks shared by every path: env sanity, all CUDA, non-COW."""
    if not _has_cutedsl() or _deterministic():
        return False
    if not all(t.is_cuda for t in tensors):
        return False
    if _any_cow(*tensors):
        return False
    return True


# ---------------------------------------------------------------------------
# TI-driven layout analysis.
#
# Mirrors aten's pattern in ScatterGatherKernel.cu / IndexKernelUtils.h:
# restride ``self`` so its scatter-axis stride is 0 and its shape matches
# ``index``, then build a TensorIterator on the three. TensorIterator does
# the broadcast / coalesce / dim reorder; we then pattern-match the post-
# build iter strides to recognize the kernel's expected layout.
# ---------------------------------------------------------------------------


def _normalize_dim(dim: int, ndim: int) -> int:
    return dim + ndim if dim < 0 else dim


def _scatter_add_eligibility(
    self: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> TensorIterator | None:
    """Return the analysis TensorIterator if (self, dim, index, src) fits
    the kernel's expected layout, else ``None``.

    Mirrors aten's ``fast_scatter_add_kernel_eligible`` (IndexKernelUtils.h):
    restride ``self`` so its scatter-axis stride is 0 (shape = ``index.shape``),
    let TensorIterator coalesce + reorder, then check that the result is
    a 2D iter where dim 0 is the contiguous slice axis and dim 1 is the
    scatter (index) axis.
    """
    if self.dtype not in _SUPPORTED_DTYPES or self.dtype != src.dtype:
        return None
    if index.dtype not in (torch.int32, torch.int64):
        return None
    if self.ndim != src.ndim or self.ndim != index.ndim or self.ndim == 0:
        return None
    d = _normalize_dim(dim, self.ndim)
    if not 0 <= d < self.ndim:
        return None

    self_strides = list(self.stride())
    self_strides[d] = 0  # aten's restride_dim trick
    try:
        self_r = self.as_strided(index.shape, self_strides)
        src_r = src.as_strided(index.shape, src.stride())
        it = (
            TensorIteratorConfig()
            .add_output(self_r)
            .add_const_input(src_r)
            .add_const_input(index)
            .set_check_mem_overlap(False)
            .check_all_same_dtype(False)
            .resize_outputs(False)
            .build()
        )
    except RuntimeError:
        return None

    if it.ndim != 2:
        return None
    elem = self.element_size()
    idx_elem = index.element_size()
    s_self, s_src, s_idx = it.strides(0), it.strides(1), it.strides(2)
    if (
        s_idx[0] == 0
        and s_idx[1] == idx_elem
        and s_self[0] == elem
        and s_src[0] == elem
        and s_self[1] == 0
    ):
        return it
    return None


# ---------------------------------------------------------------------------
# Per-kernel eligibility. ``it.shape[0]`` is the slice extent (N elements)
# after coalescing; that's what the alignment / divisibility checks key on.
# ---------------------------------------------------------------------------


def _is_tma_supported(
    self: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> bool:
    if not _has_sm90_plus():
        return False
    it = _scatter_add_eligibility(self, dim, index, src)
    if it is None:
        return False
    # TMA chunk_bytes = min(row_bytes, 512); row_bytes must evenly divide
    # by chunk_bytes and chunk_bytes must be 16-byte aligned.
    from .tma_kernel import row_shape_supported

    return row_shape_supported(self.dtype, it.shape[0])


def _is_vec_scatter_supported(
    self: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> bool:
    # ``red.global.add.noftz.bf16x2`` requires sm_90+. fp16x2 (sm_70+) and
    # scalar fp32 atomicAdd (sm_60+) work everywhere we care about.
    if self.dtype is torch.bfloat16 and not _has_sm90_plus():
        return False
    it = _scatter_add_eligibility(self, dim, index, src)
    if it is None:
        return False
    # Each lane owns vec_elems consecutive elements; loop is either
    # fully in-bounds or fully skipped per lane. Only requirement is
    # N % vec_elems == 0 (fp32: %4, halves: %8).
    from .vec_scatter_kernel import vec_elems_for

    return it.shape[0] % vec_elems_for(self.dtype) == 0


# ---------------------------------------------------------------------------
# Kernel-input prep. After eligibility the kernel host functions take 2D
# (M, N) tensors and a 1D index. We synthesize those views from the user's
# original operands using the TI-approved geometry.
# ---------------------------------------------------------------------------


def _prepare_kernel_inputs(
    self: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    it: TensorIterator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (self_2d, index_1d, src_2d) for the kernel host functions.

    Caller has verified eligibility via ``_scatter_add_eligibility``.
    ``it.shape[0]`` is the coalesced slice extent (product of every
    non-scatter dim of ``self``). The 2D views preserve the user's
    original stride along ``dim`` (used by the kernel for the row stride).
    """
    d = _normalize_dim(dim, self.ndim)
    N, M_src = it.shape[0], it.shape[1]
    M_self = self.shape[d]
    self_2d = self.as_strided((M_self, N), (self.stride(d), 1))
    src_2d = src.as_strided((M_src, N), (src.stride(d), 1))
    # ``index`` may be nD with broadcast (stride-0) axes alongside one
    # data axis. Drop every broadcast axis by selecting position 0 to
    # land on the 1D view the kernels want.
    index_1d = index
    while index_1d.ndim > 1:
        ax = next(
            (a for a in range(index_1d.ndim) if index_1d.stride(a) == 0),
            None,
        )
        if ax is None:
            raise RuntimeError(
                "scatter_add cutedsl: index lacks a broadcast axis to collapse"
            )
        index_1d = index_1d.select(ax, 0)
    if index_1d.dtype != torch.int64:
        index_1d = index_1d.to(torch.int64)
    if not index_1d.is_contiguous():
        index_1d = index_1d.contiguous()
    return self_2d, index_1d, src_2d


# ---------------------------------------------------------------------------
# Cond / impl factories. ``_make_cond`` wraps a support check in the shared
# dispatch boilerplate; ``_make_impls`` produces the (functional, .out,
# in-place) triple for a given kernel.
# ---------------------------------------------------------------------------


def _make_cond(
    support_check,
    *,
    requires_out: bool = False,
) -> _OpCondFn:
    """Build a cond function that runs ``support_check`` behind the shared
    boilerplate (cutedsl availability, non-deterministic mode, CUDA
    tensors, non-COW, dtype/shape match on ``out`` when present).

    ``support_check`` is ``_is_tma_supported`` or ``_is_vec_scatter_supported``
    -- both take ``(self, dim, index, src)`` and rebuild the analysis iter
    internally.
    """
    if requires_out:

        def _cond(self, dim, index, src, *, out):
            if not _base_cond_ok(self, index, src, out):
                return False
            if out.dtype != self.dtype or out.shape != self.shape:
                return False
            if out.ndim == 0:
                return False
            # ``out`` is the destination of a functional .out call; it
            # plays the role of self in the iter analysis.
            return support_check(out, dim, index, src)

        return _cond

    def _cond(self, dim, index, src, *args, **kwargs):
        if not _base_cond_ok(self, index, src):
            return False
        return support_check(self, dim, index, src)

    return _cond


def _copy_if_distinct(out: torch.Tensor, self: torch.Tensor) -> None:
    """Seed ``out`` with ``self``'s values when they're separate buffers.
    Cond guarantees shape match, so no resize is needed."""
    if out.data_ptr() != self.data_ptr():
        out.copy_(self)


def _make_impls(kernel_getter):
    """Build (functional, out, in-place) impl callables.

    ``kernel_getter`` lazily imports the ``*_scatter_add_into`` kernel
    host function (we defer to keep the DSL runtime out of registration).
    Cond has already verified eligibility; we rebuild the iter inside the
    impl to recover the post-coalesce geometry that ``_prepare_kernel_inputs``
    needs. The build is pure C++ and cheap.
    """

    def _run(dst: torch.Tensor, dim: int, index, src) -> torch.Tensor:
        it = _scatter_add_eligibility(dst, dim, index, src)
        if it is None:
            raise RuntimeError(
                "scatter_add cutedsl: cond approved but iter rebuild failed"
            )
        dst_2d, index_1d, src_2d = _prepare_kernel_inputs(dst, dim, index, src, it)
        kernel_getter()(dst_2d, index_1d, src_2d)
        return dst

    def impl(self, dim, index, src, *args, **kwargs):
        return _run(self.clone(), dim, index, src)

    def out_impl(self, dim, index, src, *, out):
        _copy_if_distinct(out, self)
        return _run(out, dim, index, src)

    def inplace_impl(self, dim, index, src, *args, **kwargs):
        return _run(self, dim, index, src)

    return impl, out_impl, inplace_impl


def _tma_kernel():
    from .tma_kernel import tma_scatter_add_into

    return tma_scatter_add_into


def _vs_kernel():
    from .vec_scatter_kernel import vec_scatter_add_into

    return vec_scatter_add_into


_tma_impl, _tma_out_impl, _tma_inplace_impl = _make_impls(_tma_kernel)
_vs_impl, _vs_out_impl, _vs_inplace_impl = _make_impls(_vs_kernel)

_tma_cond = _make_cond(_is_tma_supported)
_tma_out_cond = _make_cond(_is_tma_supported, requires_out=True)
_vs_cond = _make_cond(_is_vec_scatter_supported)
_vs_out_cond = _make_cond(_is_vec_scatter_supported, requires_out=True)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


# Dispatch priority is first-registered-wins. TMA's cond is strictly
# narrower than vec-scatter's (TMA only fires on sm_90+ with stricter D
# alignment), so TMA goes first.
_PATHS = (
    # (cond, impl, out_cond, out_impl, inplace_impl)
    (_tma_cond, _tma_impl, _tma_out_cond, _tma_out_impl, _tma_inplace_impl),
    (_vs_cond, _vs_impl, _vs_out_cond, _vs_out_impl, _vs_inplace_impl),
)


def _register_path(
    dispatch_key: str,
    cond: _OpCondFn,
    impl: _OpImplFn,
    out_cond: _OpCondFn,
    out_impl: _OpImplFn,
    inplace_impl: _OpImplFn,
) -> None:
    # scatter_add_ has its own dispatcher entry separate from scatter_add
    # and scatter_add.out; it's structured-delegated to .out below the
    # dispatcher layer, so overriding .out alone doesn't catch it.
    for op_symbol, cond_fn, impl_fn in (
        ("scatter_add", cond, impl),
        ("scatter_add.out", out_cond, out_impl),
        ("scatter_add_", cond, inplace_impl),
    ):
        cu.register_op_override(
            "aten",
            op_symbol,
            dispatch_key,
            cond=cond_fn,
            impl=impl_fn,
            allow_multiple_override=True,
        )


def register_to_dispatch() -> None:
    if not _has_cutedsl():
        return
    for path in _PATHS:
        _register_path("CUDA", *path)
