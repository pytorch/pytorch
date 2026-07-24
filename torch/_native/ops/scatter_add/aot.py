"""Native-AOT declaration for aten::scatter_add @ CUDA (TMA path only).

Only the TMA kernel (sm_90+, cp.reduce.async.bulk) is AOT-embedded; the
vec-scatter fallback shapes stay JIT-eligible, so covered_axes reports a
"tma" axis computed with the SAME eligibility helpers the JIT cond uses
(cutedsl_impl) and the grid pins it True. Everything about the call is
a runtime argument, so the grid is one kernel per dtype.

The C++ prelude ports the TI-driven layout analysis: restride out so its
scatter-axis stride is 0 with index's shape, let TensorIterator coalesce
and reorder, then pattern-match the 2D iter strides -- the same scheme
as aten's fast_scatter_add_kernel_eligible (IndexKernelUtils.h) and the
Python _scatter_add_eligibility. The analysis runs on ``out`` (the
tensor the kernel writes); the wrapper's meta() gives it self's shape,
and for in-place calls out IS self.

Module scope must import with stdlib alone (torchgen loads this
pre-build); torch is imported lazily inside covered_axes.
"""

ATEN_OP = "scatter_add"
DISPATCH_KEY = "CUDA"
KERNEL_MODULE = "tma_kernel.py"

# dtype -> (ScalarType, itemsize). chunk_elems (the static TMA box dim)
# is 512B / itemsize, mirroring tma_kernel.chunk_elems_for.
_DTYPES = {
    "float32": ("at::kFloat", 4),
    "float16": ("at::kHalf", 2),
    "bfloat16": ("at::kBFloat16", 2),
}


def kernel_precompile_grid():
    # One TMA kernel per dtype; shapes/strides/grid are runtime args.
    # "tma" pins coverage to TMA-eligible calls only: vec-scatter-only
    # shapes (alignment or sm < 90) report tma=False and stay JIT.
    return [{"dtype": list(_DTYPES), "tma": True, "deterministic": False}]


def _cheap_tma_covered(self, dim, index, src, out):
    """Cheap (no TensorIterator) projection of TMA eligibility.

    covered_axes runs on EVERY call of the op (it is the JIT-decline
    check), so it must not build a TensorIterator the way the JIT cond
    does -- that costs ~30us/call in Python and would erase the AOT
    path's host-overhead win. Instead pattern-match the canonical
    layout directly: dim 0 scatter, expanded-1D index (stride (1, 0,
    ...)), inner-contiguous rows on src/dst with everything past dim 0
    dense. This is deliberately NARROWER than the C++ prelude's TI
    analysis (rank-permuted layouts that TI would coalesce stay JIT --
    the JIT TMA path serves them with the same kernel); it must never
    be WIDER than TMA eligibility or vec-scatter-only calls would
    decline JIT and land unaccelerated on stock aten.
    """
    import torch

    if index.dtype != torch.int64:
        return False  # AOT ABI is int64-only; JIT widens int32
    ndim = self.dim()
    if ndim < 2 or src.dim() != ndim or index.dim() != ndim:
        return False
    d = dim + ndim if dim < 0 else dim
    if d != 0:
        return False
    if index.shape != src.shape:
        return False
    if index.stride(0) != 1 or any(index.stride(k) != 0 for k in range(1, ndim)):
        return False
    dst = self if out is None else out
    if out is not None and (out.dtype != self.dtype or out.shape != self.shape):
        return False

    # Rows must be dense past dim 0 (collapsible to 2D with row stride
    # = stride(0)): exactly the contiguous-tail check.
    def rows_dense(t):
        expect = 1
        for k in range(ndim - 1, 0, -1):
            if t.stride(k) != expect:
                return False
            expect *= t.size(k)
        return True

    if not rows_dense(src) or not rows_dense(dst):
        return False
    n = src.numel() // src.size(0)
    elem = self.element_size()
    if (n * elem) % 16 or (src.stride(0) * elem) % 16 or (dst.stride(0) * elem) % 16:
        return False
    # Alignment via storage_offset, NOT data_ptr(): Python data_ptr()
    # materializes copy-on-write inputs and coverage runs on every call.
    if (src.storage_offset() * elem) % 16 or (dst.storage_offset() * elem) % 16:
        return False
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    return torch.cuda.get_device_capability(self.device)[0] >= 9


def covered_axes(self, dim, index, src, out=None):
    import torch

    return {
        "dtype": self.dtype,
        "tma": _cheap_tma_covered(self, dim, index, src, out),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }


def cpp_covers():
    # C++ port of covered_axes + grid matching (registered by the AOT
    # library as torch.ops._native_aot.covers_scatter_add; ~1.5us vs
    # ~7us for the Python path). Same covered set as _cheap_tma_covered.
    dtype_reject = " && ".join(f"st != {t}" for t, _ in _DTYPES.values())
    return f"""
      const auto st = self.scalar_type();
      if ({dtype_reject}) return false;
      if (at::globalContext().deterministicAlgorithms()) return false;
      if (index.scalar_type() != at::kLong) return false;
      const int64_t ndim = self.dim();
      if (ndim < 2 || src.dim() != ndim || index.dim() != ndim) return false;
      const int64_t d = dim < 0 ? dim + ndim : dim;
      if (d != 0) return false;
      if (index.sizes() != src.sizes()) return false;
      if (index.stride(0) != 1) return false;
      for (int64_t k = 1; k < ndim; ++k) {{
        if (index.stride(k) != 0) return false;
      }}
      const at::Tensor& dst = out.has_value() ? *out : self;
      if (out.has_value() && (out->scalar_type() != st || out->sizes() != self.sizes())) return false;
      // Rows dense past dim 0 (collapsible to 2D with row stride = stride(0)).
      int64_t expect = 1;
      for (int64_t k = ndim - 1; k >= 1; --k) {{
        if (src.stride(k) != expect || dst.stride(k) != expect) return false;
        expect *= src.size(k);
      }}
      const int64_t elem = self.element_size();
      const int64_t n = src.numel() / std::max<int64_t>(src.size(0), 1);
      if ((n * elem) % 16 || (src.stride(0) * elem) % 16 || (dst.stride(0) * elem) % 16) return false;
      if (reinterpret_cast<uintptr_t>(src.const_data_ptr()) % 16 || reinterpret_cast<uintptr_t>(dst.const_data_ptr()) % 16) return false;
      // sm gate: emitted by gen_aot_lib from ARCHS x shipped artifacts.
      return self.is_cuda();
    """


def cpp_dispatch_prelude():
    dtype_reject = " && ".join(f"st != {t}" for t, _ in _DTYPES.values())
    return f"""
      const auto st = self.scalar_type();
      if ({dtype_reject}) return false;
      if (index.scalar_type() != at::kLong) return false;
      if (self.dim() == 0 || self.dim() != index.dim() || self.dim() != src.dim()) return false;
      if (at::globalContext().deterministicAlgorithms()) return false;
      const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
      if (index.numel() == 0) {{
        if (!out.is_same(self)) out.copy_(self);
        return true;
      }}
      // TI-driven layout analysis (same scheme as aten's
      // fast_scatter_add_kernel_eligible in IndexKernelUtils.h): restride
      // out so its scatter-axis stride is 0 with index's shape, let
      // TensorIterator coalesce + reorder, then require a 2D iter whose
      // dim 0 is the contiguous slice axis and dim 1 the index axis.
      auto out_r_strides = out.strides().vec();
      out_r_strides[d] = 0;
      auto out_r = out.as_strided(index.sizes(), out_r_strides);
      auto src_r = src.as_strided(index.sizes(), src.strides());
      auto it = at::TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(out_r)
                    .add_const_input(src_r)
                    .add_const_input(index)
                    .build();
      if (it.ndim() != 2) return false;
      const int64_t elem = out.element_size();
      if (it.strides(2)[0] != 0 || it.strides(2)[1] != index.element_size()) return false;
      if (it.strides(0)[0] != elem || it.strides(1)[0] != elem || it.strides(0)[1] != 0) return false;
      const int64_t N = it.shape()[0];
      const int64_t M_src = it.shape()[1];
      // 16B contract: TMA tile load of src rows and the
      // cp.reduce.async.bulk gmem operand on out rows.
      if ((N * elem) % 16 != 0) return false;
      if (reinterpret_cast<uintptr_t>(src.const_data_ptr()) % 16 != 0) return false;
      if (reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 != 0) return false;
      if ((src.stride(d) * elem) % 16 != 0 || (out.stride(d) * elem) % 16 != 0) return false;
    """


def cpp_dispatch(spec):
    return f"st == {_DTYPES[spec['dtype']][0]}"


def cpp_launch(spec, launch_fn):
    chunk_elems = 512 // _DTYPES[spec["dtype"]][1]
    return f"""
      if (!out.is_same(self)) out.copy_(self);
      auto out_2d = out.as_strided({{out.size(d), N}}, {{out.stride(d), 1}});
      auto src_2d = src.as_strided({{M_src, N}}, {{src.stride(d), 1}});
      auto index_1d = index.as_strided({{M_src}}, {{1}});
      const int64_t num_chunks = (N + {chunk_elems} - 1) / {chunk_elems};
      // Grid plan (mirrors tma_kernel._plan_grid): one warp per CTA, so
      // the row axis alone must reach sm*32 CTAs for occupancy; when M
      // is too small, split the chunk axis across grid_y.
      const int64_t sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
      int64_t grid_x = M_src, grid_y = 1, chunks_per_cta = num_chunks;
      if (M_src >= sm * 32) {{
        grid_x = std::min<int64_t>(M_src, sm * 64);
      }} else {{
        const int64_t want_y = std::max<int64_t>(1, (sm * 32) / std::max<int64_t>(M_src, 1));
        grid_y = std::min(num_chunks, want_y);
        chunks_per_cta = (num_chunks + grid_y - 1) / grid_y;
        grid_y = (num_chunks + chunks_per_cta - 1) / chunks_per_cta;
      }}
      {launch_fn}(src_2d, index_1d, out_2d, static_cast<int32_t>(N),
                  static_cast<int32_t>(num_chunks), static_cast<int32_t>(chunks_per_cta),
                  static_cast<int32_t>(grid_x), static_cast<int32_t>(grid_y),
                  out.stride(d), at::cuda::getCurrentCUDAStream());
    """
