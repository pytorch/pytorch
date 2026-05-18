"""CuTeDSL override registrations for ``aten::scatter_add``.

Two kernels for the ``index.unsqueeze(-1).expand(-1, ...)`` expanded-1D
pattern (same shape for self/src/index, inner-contiguous after permuting
``dim`` to axis 0), tried in order:

1. **TMA** (``tma_kernel.py``, sm_90+): uses ``cp.reduce.async.bulk`` to
   offload the whole reduction to the TMA unit, 3-7x faster than aten on
   high-contention embedding-bag-grad-style workloads.
2. **vec-scatter** (``vec_scatter_kernel.py``, any SM): warp-per-row
   scheduling with ``atomicAdd`` (scalar fp32, paired x2 halves). Covers
   pre-sm_90, and sm_90+ shapes where the TMA alignment constraint isn't
   met.

Both kernels assume the scatter axis is dim=0 and the inner dims pack
contiguously. We accept any ``dim`` whose post-``movedim(dim, 0)`` layout
satisfies that constraint -- in particular, default-contiguous tensors
with ``dim=0`` and pre-permuted views like ``x.permute(1, 0, 2)`` with
``dim=1``. ``_permute_to_dim0`` validates the permuted operand triple via
``TensorIteratorConfig`` (cross-device / shape / overlap checks) before
the path-specific support check runs.

Anything else (non-expanded index, post-permute non-contiguous inputs,
deterministic mode) falls through to aten. Aten is competitive or better
on those shapes, so we intentionally don't override them.

In-place (``scatter_add_``) is registered explicitly; PyTorch's
structured-delegate from ``scatter_add_`` to ``scatter_add.out`` happens
below the dispatcher, so overriding the ``.out`` variant alone doesn't
intercept the in-place method.
"""

import functools
import importlib.util
import math

import torch
from torch._tensor_iterator import binary_op as _ti_binary_op

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
# Expanded-1D helpers, shared by TMA and vec-scatter paths.
# ---------------------------------------------------------------------------


def _inner_contiguous(t: torch.Tensor) -> bool:
    """True iff ``t.shape[1:]`` is packed contiguously (so flattening to
    ``(t.shape[0], prod(t.shape[1:]))`` is a valid view), regardless of
    the outer stride. Covers slices like ``X[:, :K]`` of a wider buffer.

    Empty outer axis returns ``False``: the kernel has no work to do,
    and letting it through means ``select(0, 0)`` below would raise.
    """
    if t.ndim == 0:
        return False
    if t.ndim == 1:
        return t.stride(0) == 1
    if t.shape[0] == 0:
        return False
    return t.select(0, 0).is_contiguous()


def _normalize_dim(dim: int, ndim: int) -> int:
    return dim + ndim if dim < 0 else dim


def _permute_to_dim0(
    self: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Move ``dim`` to axis 0 on all three operands and validate.

    The downstream pipeline (``_expanded_1d_inner_size``,
    ``_flatten_for_expanded_1d``, the kernels) hard-codes dim=0 and
    inner-contiguous nD tensors. ``movedim(dim, 0)`` is a view, so writes
    through the permuted ``self`` view land in the correct storage cells of
    the user's original tensor.

    Per-slice validation via TensorIterator mirrors aten's
    ``index_func_meta_impl`` trick (TensorAdvancedIndexing.cpp:441-444): the
    full (self, src) pair has different sizes along axis 0, but
    ``select(0, 0)`` of each gives matching shapes that TI can sanity-check
    for cross-device, dtype, and overlap. Aten itself doesn't run TI on the
    full triple either -- it uses the dedicated ``scatter_shape_check``
    helper, which is the analog of our ``_expanded_1d_inner_size`` checks
    downstream. Permutation alone is not enough: the post-permute ``self``
    and ``src`` must still pass ``_inner_contiguous`` for the kernel's
    2D-view flattening to be valid -- that's checked downstream.
    """
    if self.ndim != src.ndim or self.ndim != index.ndim or self.ndim < 2:
        return None
    d = _normalize_dim(dim, self.ndim)
    if d < 0 or d >= self.ndim:
        return None
    if self.shape[d] == 0 or src.shape[d] == 0:
        return None
    self_p = self.movedim(d, 0)
    src_p = src.movedim(d, 0)
    index_p = index.movedim(d, 0)

    # Per-slice TensorIterator check: build a binary_op on the first slice
    # of each post-permute operand. The slice shapes match by
    # construction, so TI's broadcast / device / overlap rules apply
    # cleanly -- this catches cross-device operands and inner-shape
    # mismatches before the kernel-specific checks run.
    try:
        _ti_binary_op(self_p.select(0, 0), self_p.select(0, 0), src_p.select(0, 0))
    except RuntimeError:
        return None
    return self_p, src_p, index_p


def _expanded_1d_inner_size(
    self: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> int | None:
    """Return ``prod(src.shape[1:])`` when the ``index.unsqueeze(-1)
    .expand(-1, ...)`` fast-path pattern applies on (already permuted to
    dim=0) operands, else ``None``.

    Requirements:
      - self, src, index have the same rank (>= 2) and shape
      - self, src have inner-dim stride 1 (outer row stride can differ
        from prod(shape[1:]) -- e.g. a slice like ``X[:, :K]`` of a
        wider buffer works)
      - index.stride(i) == 0 for every axis i > 0 (the broadcast)
      - dtype in ``_SUPPORTED_DTYPES`` and self.dtype == src.dtype
      - index.dtype in {int32, int64} (int32 is cast to int64 on the
        host before the kernel launches)

    Caller is responsible for the dim=0 invariant: see ``_permute_to_dim0``.
    """
    if self.dtype not in _SUPPORTED_DTYPES or self.dtype != src.dtype:
        return None
    if index.dtype not in (torch.int32, torch.int64):
        return None
    if self.ndim < 2:
        return None
    # Inner-dim stride 1, plus ensure inner dims pack contiguously so we
    # can flatten shape[1:] into a single N. A simple sufficient check:
    # stride of each inner axis == product of strides below it.
    if not _inner_contiguous(self) or not _inner_contiguous(src):
        return None
    if index.shape != src.shape or self.shape[1:] != src.shape[1:]:
        return None
    for i in range(1, index.ndim):
        if index.stride(i) != 0:
            return None
    # shape[0] == 0 on self or src is already rejected by
    # ``_inner_contiguous`` above.
    N = math.prod(src.shape[1:])
    if N == 0:
        return None
    return N


def _flatten_2d_view(t: torch.Tensor) -> torch.Tensor:
    """Flatten ``t.shape[1:]`` into a single N. Preserves the outer row
    stride (so a slice like ``X[:, :K]`` stays a view of the wider
    buffer). Caller must have verified ``_inner_contiguous(t)``.
    """
    if t.ndim == 2:
        return t
    N = math.prod(t.shape[1:])
    # Outer stride carries through unchanged; inner dims collapse to
    # stride 1 because they were packed (cond-enforced).
    return t.as_strided((t.shape[0], N), (t.stride(0), 1))


def _flatten_for_expanded_1d(
    self: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reshape eligible nD inputs to (self_2d, index_1d, src_2d).

    Caller must have verified eligibility via ``_expanded_1d_inner_size``.
    All reshapes are views; the 1D index view is taken by selecting
    element 0 of every broadcast axis.
    """
    self_2d = _flatten_2d_view(self)
    src_2d = _flatten_2d_view(src)
    index_1d = index
    for _ in range(index.ndim - 1):
        index_1d = index_1d.select(-1, 0)
    if index_1d.dtype != torch.int64:
        # Kernel expects int64. aten accepts both int32 and int64; we
        # widen on the host so the kernel stays dtype-specialized.
        index_1d = index_1d.to(torch.int64)
    if not index_1d.is_contiguous():
        index_1d = index_1d.contiguous()
    return self_2d, index_1d, src_2d


# ---------------------------------------------------------------------------
# Path-specific eligibility checks.
# ---------------------------------------------------------------------------


def _is_tma_supported(
    self: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> bool:
    """Eligibility check on (self, index, src) ALREADY permuted to dim=0."""
    if not _has_sm90_plus():
        return False
    N = _expanded_1d_inner_size(self, index, src)
    if N is None:
        return False
    # TMA transfer size is baked in at compile time (chunk_bytes =
    # min(row_bytes, 512)); row_bytes must evenly divide by chunk_bytes
    # and chunk_bytes must be 16-byte aligned.
    from .tma_kernel import row_shape_supported

    return row_shape_supported(self.dtype, N)


def _is_vec_scatter_supported(
    self: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> bool:
    """Eligibility check on (self, index, src) ALREADY permuted to dim=0."""
    # ``red.global.add.noftz.bf16x2`` requires sm_90+. fp16x2 (sm_70+)
    # and scalar fp32 atomicAdd (sm_60+) work everywhere we care about.
    if self.dtype is torch.bfloat16 and not _has_sm90_plus():
        return False
    N = _expanded_1d_inner_size(self, index, src)
    if N is None:
        return False
    # Each lane owns vec_elems consecutive elements; the loop is either
    # fully in-bounds or fully skipped per lane. Only requirement is
    # D % vec_elems == 0 (fp32: %4, halves: %8).
    from .vec_scatter_kernel import vec_elems_for

    return N % vec_elems_for(self.dtype) == 0


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
    boilerplate (cutedsl availability, non-deterministic mode, CUDA tensors,
    non-COW, dtype/shape match on ``out`` when present).

    ``support_check`` is invoked on the operands AFTER ``movedim(dim, 0)``;
    if the permutation can't be validated by TensorIterator (cross-device,
    shape mismatch, etc.) we reject without running the path-specific
    check.
    """
    if requires_out:

        def _cond(self, dim, index, src, *, out):
            if not _base_cond_ok(self, index, src, out):
                return False
            if out.dtype != self.dtype or out.shape != self.shape:
                return False
            if out.ndim == 0:
                return False
            # ``out`` is structurally a clone of ``self`` from the user's
            # perspective; permute it the same way for the inner-contig
            # check on the post-permute view.
            permuted = _permute_to_dim0(out, dim, index, src)
            if permuted is None:
                return False
            out_p, src_p, index_p = permuted
            if not _inner_contiguous(out_p):
                return False
            return support_check(out_p, index_p, src_p)

        return _cond

    def _cond(self, dim, index, src, *args, **kwargs):
        if not _base_cond_ok(self, index, src):
            return False
        permuted = _permute_to_dim0(self, dim, index, src)
        if permuted is None:
            return False
        self_p, src_p, index_p = permuted
        return support_check(self_p, index_p, src_p)

    return _cond


def _copy_if_distinct(out: torch.Tensor, self: torch.Tensor) -> None:
    """Seed ``out`` with ``self``'s values when they're separate buffers.
    Cond guarantees shape match, so no resize is needed."""
    if out.data_ptr() != self.data_ptr():
        out.copy_(self)


def _make_impls(kernel_getter):
    """Build (functional, out, in-place) impl callables for an
    expanded-1D-pattern kernel. ``kernel_getter`` is a zero-arg callable
    that lazily imports and returns the ``*_scatter_add_into`` kernel; we
    defer the import so the DSL runtime isn't pulled in at registration
    time.

    Each impl ``movedim(dim, 0)``s the operands before kernel dispatch.
    The permuted ``dst`` is a view of the original storage, so the kernel's
    writes land in the correct cells of the user's tensor.
    """

    def _run(dst: torch.Tensor, dim: int, index, src) -> torch.Tensor:
        # Cond guarantees the permuted views pass _expanded_1d_inner_size.
        d = _normalize_dim(dim, dst.ndim)
        dst_p = dst.movedim(d, 0)
        src_p = src.movedim(d, 0)
        index_p = index.movedim(d, 0)
        dst_2d, index_1d, src_2d = _flatten_for_expanded_1d(dst_p, index_p, src_p)
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
