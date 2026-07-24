"""Native-AOT declaration for aten::index_add @ CUDA (TMA path).

index_add(x, 0, index, source) with alpha=1 IS scatter_add(x, 0,
index.unsqueeze(-1).expand_as(source), source); the builder re-exports
the scatter_add TMA kernel under this op's prefix (one kernel body, two
aten ops). Coverage delegates to scatter_add's _cheap_tma_covered on
the expanded index, so TMA eligibility has one source of truth.

Beyond kernel sharing this exercises: a structured op with a
precomputed dim (the C++ side receives the already-wrapped value;
covered_axes sees the RAW schema call and normalizes itself), a
prelude reject on a Scalar's VALUE (alpha != 1 declines -- the kernel
bakes alpha=1), and a prelude early-return (empty index -> copy only).
There is no JIT override for index_add: uncovered calls land on stock
aten.

Module scope must import with stdlib alone (torchgen loads this
pre-build); torch and the scatter_add declaration are loaded lazily.
"""

import functools
import importlib.util
import os


ATEN_OP = "index_add"
DISPATCH_KEY = "CUDA"
KERNEL_MODULE = "aot_kernel.py"

_DTYPES = {
    "float32": ("at::kFloat", 4),
    "float16": ("at::kHalf", 2),
    "bfloat16": ("at::kBFloat16", 2),
}


@functools.cache
def _scatter_add_aot():
    # File-path load (no package context at torchgen time); the sibling
    # declaration owns _cheap_tma_covered, the TMA-eligibility check.
    # Cached: covered_axes calls this and exec_module costs ~80us.
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "scatter_add", "aot.py")
    spec = importlib.util.spec_from_file_location("scatter_add_aot_for_index_add", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def kernel_precompile_grid():
    # Same grid as scatter_add: one TMA kernel per dtype, all shapes
    # runtime. "tma" pins coverage to TMA-eligible calls.
    return [{"dtype": list(_DTYPES), "tma": True, "deterministic": False}]


# No cpp_covers, deliberately: index_add has no JIT override, so the
# router never consults coverage per call -- a C++ covers op would be
# dead weight. covered_axes exists for the contract (and tooling/tests).
def covered_axes(self, dim, index, source, alpha=1):
    import torch

    # index_add's index is 1D; the TMA layout contract is stated over
    # the scatter_add form, so check eligibility on the expanded view
    # (free, no copy). alpha != 1 stays with stock aten: the kernel is
    # a pure add (alpha baked at 1).
    tma = False
    if (
        alpha == 1
        and isinstance(index, torch.Tensor)
        and index.dim() == 1
        and index.is_contiguous()
        and isinstance(source, torch.Tensor)
        and source.dim() == self.dim()
        and source.dim() >= 2
        and index.numel() == source.shape[0]
    ):
        expanded = index.view(-1, *([1] * (source.dim() - 1))).expand_as(source)
        tma = _scatter_add_aot()._cheap_tma_covered(self, dim, expanded, source, None)
    return {
        "dtype": self.dtype,
        "tma": tma,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }


def cpp_dispatch_prelude():
    dtype_reject = " && ".join(f"st != {t}" for t, _ in _DTYPES.values())
    # dim arrives precomputed (maybe_wrap_dim applied by the structured
    # META), unlike the raw schema dim covered_axes sees. index is 1D by
    # schema, so the layout checks are direct (no TI analysis needed):
    # rows dense past dim 0 and 16B-aligned, the TMA contract.
    return f"""
      const auto st = self.scalar_type();
      if ({dtype_reject}) return false;
      if (dim != 0) return false;
      if (self.dim() < 2 || index.dim() != 1) return false;
      if (index.scalar_type() != at::kLong || !index.is_contiguous()) return false;
      if (source.scalar_type() != st || source.dim() != self.dim()) return false;
      if (!alpha.equal(1)) return false;
      if (at::globalContext().deterministicAlgorithms()) return false;
      if (index.numel() == 0) {{
        if (!out.is_same(self)) out.copy_(self);
        return true;
      }}
      if (index.numel() != source.size(0)) return false;
      // Rows dense past dim 0 (collapsible to 2D with row stride =
      // stride(0)) on both source and out.
      const auto rows_dense = [](const at::Tensor& t) {{
        int64_t expect = 1;
        for (int64_t k = t.dim() - 1; k >= 1; --k) {{
          if (t.stride(k) != expect) return false;
          expect *= t.size(k);
        }}
        return true;
      }};
      if (!rows_dense(source) || !rows_dense(out)) return false;
      const int64_t N = source.numel() / source.size(0);
      const int64_t M_src = source.size(0);
      const int64_t elem = self.element_size();
      // 16B contract: TMA tile load of source rows and the
      // cp.reduce.async.bulk gmem operand on out rows.
      if ((N * elem) % 16 != 0) return false;
      if ((source.stride(0) * elem) % 16 != 0 || (out.stride(0) * elem) % 16 != 0) return false;
      if (reinterpret_cast<uintptr_t>(source.const_data_ptr()) % 16 != 0) return false;
      if (reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 != 0) return false;
    """


def cpp_dispatch(spec):
    return f"st == {_DTYPES[spec['dtype']][0]}"


def cpp_launch(spec, launch_fn):
    chunk_elems = 512 // _DTYPES[spec["dtype"]][1]
    # Same grid plan as scatter_add's cpp_launch (mirrors
    # tma_kernel._plan_grid): one warp per CTA; when M_src alone cannot
    # fill a wave, split the chunk axis across grid_y.
    return f"""
      if (!out.is_same(self)) out.copy_(self);
      auto out_2d = out.as_strided({{out.size(0), N}}, {{out.stride(0), 1}});
      auto source_2d = source.as_strided({{M_src, N}}, {{source.stride(0), 1}});
      const int64_t num_chunks = (N + {chunk_elems} - 1) / {chunk_elems};
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
      {launch_fn}(source_2d, index, out_2d, static_cast<int32_t>(N),
                  static_cast<int32_t>(num_chunks), static_cast<int32_t>(chunks_per_cta),
                  static_cast<int32_t>(grid_x), static_cast<int32_t>(grid_y),
                  out.stride(0), at::cuda::getCurrentCUDAStream());
    """
