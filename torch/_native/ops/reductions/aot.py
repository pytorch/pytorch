"""Native-AOT declarations for CUDA reduction callables.

The family covers reductions whose TensorIterator view is rows: one unit-stride
reduced run and at most one kept run at a pitch of at least N. Each exported
callable owns its complete path: row callables launch once, while cross-CTA
callables sequence partial and final kernels. Runtime row lengths map to a small
launch-class grid, so adding sizes does not add artifacts.

Module scope must stay torch-free because torchgen loads declarations before
torch is built.
"""

import importlib.util
import os


def _load_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_aot_data.py")
    spec = importlib.util.spec_from_file_location("reduction_aot_data", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reduction AOT metadata from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DATA = _load_data()
ARCHS = _DATA.ARCHS
CORE_OPS = _DATA.CORE_OPS
DTYPES = _DATA.DTYPES
ROW_CLASSES = _DATA.ROW_CLASSES
RAGGED_CLASSES = _DATA.RAGGED_CLASSES


_DTYPE_CPP = {
    "float16": "at::kHalf",
    "bfloat16": "at::kBFloat16",
    "float32": "at::kFloat",
}
# The generated C++ reads these by name, so they stay here rather than come from the
# torch-importing modules that own them (inner_tree_plan, cutedsl_impl).
_INNER_TREE_ENV = "PYTORCH_NATIVE_INNER_TREE"
_SUM_INNER_TREE_ENV = "PYTORCH_SUM_INNER_TREE"


def _out_dtypes(dtype, has_dtype):
    # Like the JIT overrides, dtype= may only keep the input dtype or widen 16-bit
    # inputs to FP32; ATen casts the input before reducing for any other request.
    if has_dtype and dtype != "float32":
        return (dtype, "float32")
    return (dtype,)


def _red_axes(self, dim, style):
    """The reduced axes, or None for an invalid spelling; empty lists reduce all."""
    ndim = self.dim()
    if ndim < 1:
        return None
    if style == "int":
        values = [dim]
    else:
        values = [] if dim is None else [dim] if isinstance(dim, int) else list(dim)
        if not values:
            return set(range(ndim))
    if not all(isinstance(v, int) and -ndim <= v < ndim for v in values):
        return None
    axes = {v % ndim for v in values}
    return axes if len(axes) == len(values) else None


def _ti_geometry(self, dim, style):
    """(N, M, pitch) of TensorIterator's row view (kernel_general._ti_rows), or None."""
    from torch._native.ops.reductions import kernel_general as kg

    axes = _red_axes(self, dim, style)
    if axes is None or self.numel() == 0:
        return None
    red, kept = kg._ti_pairs(self, kg._probe(self, axes))
    rows = kg._ti_rows(self, axes, red, kept, any_order=True)
    if rows is None:
        return None
    M, N = rows.shape
    return (N, M, rows.stride(0) if M > 1 else N)


def _geometry_cpp(style):
    """The reduced-axis mask, then _naot_reduction_ti's row view."""
    if style == "int":
        return """
      if (self.dim() < 1 || self.dim() > 64) return false;
      const uint64_t red_mask = uint64_t{1} << c10::maybe_wrap_dim(dim, self.dim());
      int64_t N = 0, M = 0, pitch = 0;
      if (!_naot_reduction_ti(self, red_mask, N, M, pitch)) return false;
    """
    dims = "dim" if style == "list" else "*dim"
    reduce_all = (
        "dim.empty()" if style == "list" else "!dim.has_value() || dim->empty()"
    )
    return f"""
      if (self.dim() < 1 || self.dim() > 64) return false;
      uint64_t red_mask = 0;
      if ({reduce_all}) {{
        red_mask = self.dim() == 64 ? ~uint64_t{{0}} : (uint64_t{{1}} << self.dim()) - 1;
      }} else {{
        for (const auto raw : {dims}) {{
          const uint64_t bit = uint64_t{{1}} << c10::maybe_wrap_dim(raw, self.dim());
          if (red_mask & bit) return false;
          red_mask |= bit;
        }}
      }}
      int64_t N = 0, M = 0, pitch = 0;
      if (!_naot_reduction_ti(self, red_mask, N, M, pitch)) return false;
    """


def _classifier_cpp(output):
    # covers_<op> does not read ragged_tpr, which -Wunused would reject there.
    ragged_tpr = (
        "const int64_t ragged_tpr = threads_per_row;  // the ladder: use_row is false"
        if output
        else ""
    )
    return (
        """
      if (N <= 0 || self.numel() <= 0) return false;
      if (M <= 0 || M >= (int64_t{1} << 31) || N >= (int64_t{1} << 31)) return false;
      // TI's row view covers every layout; nothing kept is one flat row.
      const bool reduce_all = M == 1;
      const int64_t itemsize = self.element_size();
      const int64_t vec = 16 / itemsize;
      const bool whole = N % vec == 0 && pitch % vec == 0;

      const int64_t chunks = N / vec;
      const bool narrow = N <= 256 &&
          ((M >= 65536 && chunks <= 32) ||
           (M >= 16384 && chunks <= 16) ||
           (M >= 4096 && chunks <= 6));
      bool use_row = whole && N * itemsize <= 192 * 1024 && !(narrow && !reduce_all);

      int64_t threads_per_row = 0;
      int64_t threads_per_block = 0;
      if (N * itemsize >= 16 * 1024) {
        threads_per_row = 256;
        threads_per_block = 256;
      } else {
        threads_per_row =
            N <= 3072 ? 32 : N <= 6144 ? 64 : N <= 16384 ? 128 : 256;
        threads_per_block = N <= 16384 ? 128 : 256;
        if (threads_per_block < threads_per_row) threads_per_block = threads_per_row;
      }
      if (use_row && reduce_all) {
        const int64_t feedable = N / vec < 256 ? N / vec : 256;
        const int64_t wider =
            feedable >= 256 ? 256 : feedable >= 128 ? 128 : feedable >= 64 ? 64 : 32;
        if (wider > threads_per_row) {
          threads_per_row = wider;
          threads_per_block = wider;
        }
      }
      if (use_row &&
          (N + threads_per_row * vec - 1) / (threads_per_row * vec) > 65) {
        use_row = false;
      }
      const int64_t partials = N / @SUBROW@;
      const bool use_xcta = !use_row && whole && pitch == N &&
          N >= @MIN_XCTA@ &&
          N % @SUBROW@ == 0 &&
          partials <= @MAX_PARTIALS@ &&
          M * partials < (int64_t{1} << 31);
      // One runtime-width ragged kernel per launch class (_aot_data.ragged_class).
      const int64_t pitch_bytes = pitch * itemsize, row_bytes = N * itemsize;
      const bool late = N % 2 == 0 &&
          std::min<int64_t>(16, pitch_bytes & -pitch_bytes) >=
              std::min<int64_t>(16, row_bytes & -row_bytes);
      const int64_t ragged_min = late ? @RAGGED_EVEN_MIN@ : @RAGGED_MIN@;
      // Single rows take the JIT's reduce_all instead.
      const bool use_ragged = !use_row && !use_xcta && !whole && M > 1 &&
          N >= ragged_min && N <= @RAGGED_MAX@;
      @RAGGED_TPR@
      if (!use_row && !use_xcta && !use_ragged) return false;
    """.replace("@RAGGED_TPR@", ragged_tpr)
        .replace("@RAGGED_MIN@", str(int(_DATA.RAGGED_MIN_N)))
        .replace("@RAGGED_EVEN_MIN@", str(int(_DATA.RAGGED_EVEN_MIN_N)))
        .replace("@RAGGED_MAX@", str(int(_DATA.RAGGED_MAX_N)))
        .replace("@SUBROW@", str(int(_DATA.XCTA_SUBROW)))
        .replace("@MIN_XCTA@", str(2 * int(_DATA.XCTA_SUBROW)))
        .replace("@MAX_PARTIALS@", str(_DATA._XCTA_MAX_PARTIALS))
    )


class _RowReduction:
    DISPATCH_KEY = "CUDA"
    KERNEL_MODULE = "aot_kernel.py"
    ARCHS = ARCHS

    def __init__(self, aten_op, trait, has_dtype, dim_style):
        self.ATEN_OP = aten_op
        self._trait = trait
        self._has_dtype = has_dtype
        self._dim_style = dim_style
        # There are no inner-tree callables, so calls decline while a variable that
        # orders them on the JIT route is set: cutedsl_impl's adapter also orders sum
        # and prod under PYTORCH_SUM_INNER_TREE.
        self._order_envs = (_INNER_TREE_ENV,)
        if trait in ("sum", "prod"):
            self._order_envs += (_SUM_INNER_TREE_ENV,)

    def kernel_precompile_grid(self):
        points = []
        for dtype in DTYPES:
            for out_dtype in _out_dtypes(dtype, self._has_dtype):
                for row_class in ROW_CLASSES:
                    points.append(
                        {
                            "path": "row",
                            "trait": self._trait,
                            "dtype": dtype,
                            "out_dtype": out_dtype,
                            "row_class": row_class,
                        }
                    )
                for ragged_class in RAGGED_CLASSES:
                    points.append(
                        {
                            "path": "ragged",
                            "trait": self._trait,
                            "dtype": dtype,
                            "out_dtype": out_dtype,
                            "row_class": ragged_class,
                        }
                    )
                points.append(
                    {
                        "path": "xcta",
                        "trait": self._trait,
                        "dtype": dtype,
                        "out_dtype": out_dtype,
                    }
                )
        return points

    def covered_axes(self, self_t, dim=None, keepdim=False, *, dtype=None):
        import torch
        from torch._native.ops.reductions import kernel_rowtile as rt
        from torch._native.utils.tensor import const_data_ptr

        names = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }
        dtype_name = names.get(self_t.dtype)
        out_dtype = self_t.dtype if not self._has_dtype or dtype is None else dtype
        out_dtype_name = names.get(out_dtype)
        geometry = (
            _ti_geometry(self_t, dim, self._dim_style) if self_t.is_cuda else None
        )
        # cutedsl_impl's parse of PYTORCH_SUM_INNER_TREE, read per call as in the C++.
        sum_order = os.environ.get(_SUM_INNER_TREE_ENV, "") not in ("", "0")
        ordered = rt.inner_tree_order_enabled() or (
            _SUM_INNER_TREE_ENV in self._order_envs and sum_order
        )
        row_class = None
        path = None
        if geometry is not None and dtype_name is not None and not ordered:
            n, m, pitch = geometry
            vec = 16 // _DATA.DTYPE_BYTES[dtype_name]
            whole = n % vec == 0 and pitch % vec == 0
            row_class = _DATA.row_class(dtype_name, n, m, m == 1) if whole else None
            if row_class is not None:
                path = "row"
            elif whole and pitch == n and _DATA.xcta_ok(dtype_name, n, m):
                path = "xcta"
            else:
                row_class = _DATA.ragged_class(dtype_name, n, m, pitch)
                path = "ragged" if row_class is not None else None
        eligible = (
            self_t.is_cuda
            and dtype_name is not None
            and out_dtype_name is not None
            and const_data_ptr(self_t) % 16 == 0
            and path is not None
        )
        return {
            "path": path if eligible else None,
            "trait": self._trait,
            "dtype": self_t.dtype,
            "out_dtype": out_dtype,
            "row_class": row_class,
        }

    def cpp_helpers(self):
        return (
            _TI_HELPER_CPP
            + """
inline bool _naot_reduction_env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\\0' &&
      !(value[0] == '0' && value[1] == '\\0');
}
"""
        )

    def _front(self, *, output):
        dtype_accept = " || ".join(f"st == {_DTYPE_CPP[dtype]}" for dtype in DTYPES)
        # The input dtype, or FP32 from a 16-bit input (see _out_dtypes).
        out_dtype_accept = (
            "out_st == st || (out_st == at::kFloat && "
            "(st == at::kHalf || st == at::kBFloat16))"
        )
        if output:
            dtype_arg = (
                "if (dtype.has_value() && *dtype != out_st) return false;"
                if self._has_dtype
                else ""
            )
            out = f"""
      const auto out_st = out.scalar_type();
      if (!({out_dtype_accept})) return false;
      {dtype_arg}
      if (!out.is_contiguous()) return false;
      if (reinterpret_cast<uintptr_t>(out.mutable_data_ptr()) % 16 != 0) return false;
    """
        else:
            # Without dtype=, an out= sets the result dtype, as in the dispatch prelude.
            out_st = (
                "dtype.value_or(out.has_value() ? out->scalar_type() : st)"
                if self._has_dtype
                else "st"
            )
            out = f"""
      const auto out_st = {out_st};
      if (!({out_dtype_accept})) return false;
      if (out.has_value() &&
          (out->scalar_type() != out_st || !out->is_contiguous() ||
           reinterpret_cast<uintptr_t>(out->const_data_ptr()) % 16 != 0)) return false;
    """
        ordered = " || ".join(
            f'_naot_reduction_env_enabled("{name}")' for name in self._order_envs
        )
        return f"""
      if ({ordered}) return false;
      const auto st = self.scalar_type();
      if (!({dtype_accept})) return false;
      if (self.dim() < 1) return false;
      if (reinterpret_cast<uintptr_t>(self.const_data_ptr()) % 16 != 0) return false;
      {out}
      {_geometry_cpp(self._dim_style)}
      {_classifier_cpp(output)}
    """

    def cpp_covers(self):
        return f"{self._front(output=False)}\n      return true;"

    def cpp_dispatch_prelude(self):
        return self._front(output=True)

    def cpp_dispatch(self, spec):
        common = (
            f"st == {_DTYPE_CPP[spec['dtype']]} && "
            f"out_st == {_DTYPE_CPP[spec['out_dtype']]}"
        )
        if spec["path"] == "xcta":
            return f"{common} && use_xcta"
        if spec["path"] == "ragged":
            tpr, _ = RAGGED_CLASSES[spec["row_class"]]
            return f"{common} && use_ragged && ragged_tpr == {tpr}"
        threads_per_row, threads_per_block = ROW_CLASSES[spec["row_class"]]
        return (
            f"{common} && use_row && "
            f"threads_per_row == {threads_per_row} && "
            f"threads_per_block == {threads_per_block}"
        )

    def cpp_launch(self, spec, launch_fn):
        if spec["path"] == "xcta":
            subrow = int(_DATA.XCTA_SUBROW)
            return f"""
      const int32_t part_count = static_cast<int32_t>(partials);
      const int32_t rows = static_cast<int32_t>(M * partials);
      auto sub = self.as_strided({{rows, {subrow}}}, {{{subrow}, 1}});
      auto part = self.new_empty({{rows}}, self.options().dtype(at::kFloat));
      auto out_1d = out.view({{M}});
      {launch_fn}(sub, part, out_1d, rows, part_count, static_cast<int32_t>(M), N,
                  at::cuda::getCurrentCUDAStream());
    """
        if spec["path"] == "ragged":
            return f"""
      auto self_2d = self.as_strided({{M, N}}, {{pitch, 1}});
      auto out_1d = out.view({{M}});
      {launch_fn}(self_2d, out_1d, static_cast<int32_t>(N),
                  at::cuda::getCurrentCUDAStream());
    """
        vec = 16 // _DATA.DTYPE_BYTES[spec["dtype"]]
        threads_per_row, _ = ROW_CLASSES[spec["row_class"]]
        return f"""
      auto self_2d = self.as_strided({{M, N}}, {{pitch, 1}});
      auto out_1d = out.view({{M}});
      const int32_t nchunks = static_cast<int32_t>(N / {vec});
      const int32_t nwaves = (nchunks + {threads_per_row - 1}) / {threads_per_row};
      {launch_fn}(self_2d, out_1d, nchunks, nwaves, static_cast<int32_t>(N),
                  at::cuda::getCurrentCUDAStream());
    """


# TensorIterator's coalescing for one input and a contiguous output, without building an
# iterator (covers runs before outputs exist): reduced dims merge into one run in stride
# order and kept dims in logical (= output) order, at a pitch of at least N. Python's covered_axes
# uses the real iterator (kernel_general._ti_rows); the AOT traits have no indices, so
# reduced axes may merge in any order.
_TI_HELPER_CPP = """
inline bool _naot_reduction_ti(const at::Tensor& self, uint64_t red_mask,
                               int64_t& N, int64_t& M, int64_t& pitch) {
  const int64_t nd = self.dim();
  const auto sizes = self.sizes();
  const auto strides = self.strides();
  int64_t rs[64], rt[64];
  int nr = 0;
  for (int64_t i = 0; i < nd; ++i) {
    if (((red_mask >> i) & 1) && sizes[i] != 1) {
      rs[nr] = sizes[i];
      rt[nr] = strides[i];
      ++nr;
    }
  }
  for (int i = 1; i < nr; ++i) {
    for (int j = i; j > 0 && rt[j] < rt[j - 1]; --j) {
      std::swap(rs[j], rs[j - 1]);
      std::swap(rt[j], rt[j - 1]);
    }
  }
  N = 1;
  for (int i = 0; i < nr; ++i) {
    if (rt[i] != N) return false;
    N *= rs[i];
  }
  M = 1;
  pitch = 0;
  bool kept = false;
  for (int64_t i = nd - 1; i >= 0; --i) {
    if (((red_mask >> i) & 1) || sizes[i] == 1) continue;
    if (!kept) {
      M = sizes[i];
      pitch = strides[i];
      kept = true;
    } else if (strides[i] == M * pitch) {
      M *= sizes[i];
    } else {
      return false;
    }
  }
  if (!kept) pitch = N;
  return pitch >= N;  // overlapping (e.g. expanded) rows decline, as in _ti_rows
}
"""


def declarations():
    return [_RowReduction(*row) for row in CORE_OPS]
