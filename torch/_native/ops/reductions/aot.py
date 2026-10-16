"""Native-AOT declarations for CUDA reduction callables.

The family covers contiguous trailing-dimension reductions. Each exported
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
INDEX_OPS = _DATA.INDEX_OPS
DTYPES = _DATA.DTYPES
ROW_CLASSES = _DATA.ROW_CLASSES
ORDER_SENSITIVE_TRAITS = _DATA.ORDER_SENSITIVE_TRAITS


_DTYPE_CPP = {
    "float16": "at::kHalf",
    "bfloat16": "at::kBFloat16",
    "float32": "at::kFloat",
    "int64": "at::kLong",
}
_INNER_TREE_ENVS = ("PYTORCH_NATIVE_INNER_TREE", "PYTORCH_SUM_INNER_TREE")


def _row_geometry(self, dim, style):
    ndim = self.dim()
    if ndim < 1:
        return None
    reduce_all = False
    if style == "int":
        values = [dim]
    else:
        values = [] if dim is None else [dim] if isinstance(dim, int) else list(dim)
        reduce_all = not values
    if reduce_all:
        n = self.numel()
        return (n, 1, True) if n else None
    if not values or not all(isinstance(value, int) for value in values):
        return None
    if not all(-ndim <= value < ndim for value in values):
        return None
    dims = {value % ndim for value in values}
    if len(dims) != len(values):
        return None
    first = ndim - len(dims)
    if dims != set(range(first, ndim)):
        return None
    n = 1
    for axis in range(first, ndim):
        n *= self.shape[axis]
    return (n, self.numel() // n, False) if n and self.numel() else None


def _geometry_cpp(style):
    if style == "int":
        return """
      bool row_dims = false;
      const bool reduce_all = false;
      int64_t N = 0;
      if (self.dim() > 0) {
        const int64_t wrapped = c10::maybe_wrap_dim(dim, self.dim());
        row_dims = wrapped == self.dim() - 1;
        if (row_dims) N = self.size(wrapped);
      }
    """

    if style == "optional_int":
        return """
      const bool reduce_all = !dim.has_value();
      bool row_dims = self.dim() > 0;
      int64_t N = 0;
      if (reduce_all) {
        N = self.numel();
      } else if (row_dims) {
        const int64_t wrapped = c10::maybe_wrap_dim(*dim, self.dim());
        row_dims = wrapped == self.dim() - 1;
        if (row_dims) N = self.size(wrapped);
      }
    """

    dims = "dim" if style == "list" else "*dim"
    reduce_all = (
        "dim.empty()" if style == "list" else "!dim.has_value() || dim->empty()"
    )
    return f"""
      bool row_dims = self.dim() > 0;
      const bool reduce_all = {reduce_all};
      int64_t N = 0;
      if (reduce_all) {{
        N = self.numel();
      }} else {{
        const auto dims = {dims};
        const int64_t k = static_cast<int64_t>(dims.size());
        row_dims = row_dims && k > 0 && k <= self.dim();
        const int64_t first = self.dim() - k;
        for (int64_t axis = first; row_dims && axis < self.dim(); ++axis) {{
          bool found = false;
          for (const auto raw : dims) {{
            if (c10::maybe_wrap_dim(raw, self.dim()) == axis) found = true;
          }}
          row_dims = found;
        }}
        if (row_dims) {{
          N = 1;
          for (int64_t axis = first; axis < self.dim(); ++axis) N *= self.size(axis);
        }}
      }}
    """


def _classifier_cpp():
    return (
        """
      if (!row_dims || N <= 0 || self.numel() <= 0) return false;
      const int64_t M = self.numel() / N;
      if (M <= 0 || M >= (int64_t{1} << 31) || N >= (int64_t{1} << 31)) return false;
      const int64_t itemsize = self.element_size();
      const int64_t vec = 16 / itemsize;
      if (N % vec != 0) return false;

      const int64_t chunks = N / vec;
      const bool narrow = N <= 256 &&
          ((M >= 65536 && chunks <= 32) ||
           (M >= 16384 && chunks <= 16) ||
           (M >= 4096 && chunks <= 6));
      bool use_row = N * itemsize <= 192 * 1024 && !(narrow && !reduce_all);

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
      const bool use_xcta = !use_row &&
          N >= @MIN_XCTA@ &&
          N % @SUBROW@ == 0 &&
          partials <= @MAX_PARTIALS@ &&
          M * partials < (int64_t{1} << 31);
      if (!use_row && !use_xcta) return false;
    """.replace("@SUBROW@", str(int(_DATA.XCTA_SUBROW)))
        .replace("@MIN_XCTA@", str(2 * int(_DATA.XCTA_SUBROW)))
        .replace("@MAX_PARTIALS@", str(1 << 22))
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

    def kernel_precompile_grid(self):
        points = []
        for dtype in DTYPES:
            out_dtypes = DTYPES if self._has_dtype else (dtype,)
            for out_dtype in out_dtypes:
                for row_class in ROW_CLASSES:
                    points.append(
                        {
                            "path": "row",
                            "trait": self._trait,
                            "dtype": dtype,
                            "out_dtype": out_dtype,
                            "row_class": row_class,
                            "order": "linear",
                        }
                    )
                points.append(
                    {
                        "path": "xcta",
                        "trait": self._trait,
                        "dtype": dtype,
                        "out_dtype": out_dtype,
                        "order": "linear",
                    }
                )
                if self._trait in ORDER_SENSITIVE_TRAITS:
                    for path in ("itree_multirow", "itree_looped"):
                        for loads in _DATA.INNER_TREE_LOADS:
                            points.append(
                                {
                                    "path": path,
                                    "trait": self._trait,
                                    "dtype": dtype,
                                    "out_dtype": out_dtype,
                                    "loads": loads,
                                    "order": "inner_tree",
                                }
                            )
                    points.append(
                        {
                            "path": "itree_split",
                            "trait": self._trait,
                            "dtype": dtype,
                            "out_dtype": out_dtype,
                            "loads": 2 if _DATA.DTYPE_BYTES[dtype] == 2 else 4,
                            "order": "inner_tree",
                        }
                    )
        return points

    def covered_axes(self, self_t, dim=None, keepdim=False, *, dtype=None):
        import torch

        names = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }
        dtype_name = names.get(self_t.dtype)
        out_dtype = self_t.dtype if not self._has_dtype or dtype is None else dtype
        out_dtype_name = names.get(out_dtype)
        geometry = _row_geometry(self_t, dim, self._dim_style)
        ordered = self._trait in ORDER_SENSITIVE_TRAITS and any(
            os.environ.get(name, "") not in ("", "0") for name in _INNER_TREE_ENVS
        )
        row_class = None
        loads = None
        path = None
        if geometry is not None and dtype_name is not None:
            n, m, reduce_all = geometry
            row_class = _DATA.row_class(dtype_name, n, m, reduce_all)
            if row_class is not None:
                path = "row"
            elif _DATA.xcta_ok(dtype_name, n, m):
                path = "xcta"
            if ordered and path is not None:
                spec = _DATA.inner_tree_spec(dtype_name, n, m)
                if spec is None:
                    path = None
                else:
                    path, loads = spec
                    row_class = None
        eligible = (
            self_t.is_cuda
            and self_t.is_contiguous()
            and dtype_name is not None
            and out_dtype_name is not None
            and self_t.storage_offset() * self_t.element_size() % 16 == 0
            and path is not None
        )
        return {
            "path": path if eligible else None,
            "trait": self._trait,
            "dtype": self_t.dtype,
            "out_dtype": out_dtype,
            "row_class": row_class,
            "loads": loads,
            "order": "inner_tree" if ordered else "linear",
        }

    def cpp_helpers(self):
        return """
inline bool _naot_reduction_env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\\0' &&
      !(value[0] == '0' && value[1] == '\\0');
}

inline int64_t _naot_previous_power_of_2(int64_t value) {
  int64_t result = 1;
  while (result <= value / 2) result *= 2;
  return result;
}

inline int64_t _naot_next_power_of_2(int64_t value) {
  int64_t result = 1;
  while (result < value) result *= 2;
  return result;
}

struct _NaotInnerTreePlan {
  int64_t shape;
  int64_t loads;
  int64_t warps;
  int64_t batches;
};

inline _NaotInnerTreePlan _naot_inner_tree_plan(
    int64_t n, int64_t m, int64_t itemsize) {
  constexpr int64_t warp = 32;
  constexpr int64_t threshold = 8192;
  const int64_t vec = 16 / itemsize;
  const int64_t vectors = (n + vec - 1) / vec;
  if (vectors <= 8) {
    return {1, _naot_next_power_of_2(vectors), 1, 1};
  }

  const int64_t wle = warp * vec;
  int64_t warps = 1;
  if (n > threshold) {
    const int64_t estimated_batches = (n + threshold - 1) / threshold;
    if (estimated_batches <= 3) {
      int64_t ideal_warps = 0;
      for (int64_t b = 0; b < estimated_batches; ++b) {
        const int64_t elements = std::min(threshold, n - b * threshold);
        ideal_warps += std::min<int64_t>(16, std::max<int64_t>(1, elements / wle));
      }
      warps = std::max<int64_t>(1, ideal_warps / estimated_batches);
    } else {
      warps = std::min<int64_t>(16, std::max<int64_t>(1, threshold / wle));
    }
  } else {
    warps = std::min<int64_t>(16, std::max<int64_t>(1, n / wle));
  }
  warps = _naot_previous_power_of_2(warps);

  int64_t loads = _naot_next_power_of_2((n + warps * wle - 1) / (warps * wle));
  const int64_t max_loads =
      _naot_previous_power_of_2(std::max<int64_t>(1, threshold / (wle * warps)));
  loads = std::min(loads, max_loads);
  const int64_t batch_elements = loads * wle * warps;
  const int64_t batches = (n + batch_elements - 1) / batch_elements;
  if (batches <= 3) return {2, loads, warps, batches};
  if (batches > (std::numeric_limits<int32_t>::max() / m)) return {0, 0, 0, 0};
  return {3, loads, warps, batches};
}
"""

    def _front(self, *, output):
        dtype_accept = " || ".join(f"st == {_DTYPE_CPP[dtype]}" for dtype in DTYPES)
        out_dtype_accept = " || ".join(
            f"out_st == {_DTYPE_CPP[dtype]}" for dtype in DTYPES
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
            out_st = "dtype.value_or(st)" if self._has_dtype else "st"
            out = f"""
      const auto out_st = {out_st};
      if (!({out_dtype_accept})) return false;
      if (out.has_value() &&
          (out->scalar_type() != out_st || !out->is_contiguous() ||
           reinterpret_cast<uintptr_t>(out->const_data_ptr()) % 16 != 0)) return false;
    """
        inner_tree = (
            '_naot_reduction_env_enabled("PYTORCH_NATIVE_INNER_TREE") || '
            '_naot_reduction_env_enabled("PYTORCH_SUM_INNER_TREE")'
            if self._trait in ORDER_SENSITIVE_TRAITS
            else "false"
        )
        return f"""
      const auto st = self.scalar_type();
      if (!({dtype_accept})) return false;
      if (self.dim() < 1 || !self.is_contiguous()) return false;
      if (reinterpret_cast<uintptr_t>(self.const_data_ptr()) % 16 != 0) return false;
      {out}
      {_geometry_cpp(self._dim_style)}
      {_classifier_cpp()}
      const bool inner_tree = {inner_tree};
      const auto itree = inner_tree
          ? _naot_inner_tree_plan(N, M, self.element_size())
          : _NaotInnerTreePlan{{0, 0, 0, 0}};
      if (inner_tree && itree.shape == 0) return false;
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
        if spec["order"] == "inner_tree":
            shapes = {"itree_multirow": 1, "itree_looped": 2, "itree_split": 3}
            return (
                f"{common} && inner_tree && itree.shape == {shapes[spec['path']]} && "
                f"itree.loads == {spec['loads']}"
            )
        common = f"{common} && !inner_tree"
        if spec["path"] == "xcta":
            return f"{common} && use_xcta"
        threads_per_row, threads_per_block = ROW_CLASSES[spec["row_class"]]
        return (
            f"{common} && use_row && "
            f"threads_per_row == {threads_per_row} && "
            f"threads_per_block == {threads_per_block}"
        )

    def cpp_launch(self, spec, launch_fn):
        if spec["order"] == "inner_tree":
            common = """
      auto self_2d = self.view({M, N});
      auto out_1d = out.view({M});
"""
            if spec["path"] == "itree_multirow":
                return (
                    common
                    + f"""
      {launch_fn}(self_2d, out_1d, static_cast<int32_t>(N),
                  static_cast<int32_t>(M), at::cuda::getCurrentCUDAStream());
"""
                )
            if spec["path"] == "itree_looped":
                return (
                    common
                    + f"""
      {launch_fn}(self_2d, out_1d, static_cast<int32_t>(N),
                  static_cast<int32_t>(M), static_cast<int32_t>(itree.warps),
                  static_cast<int32_t>(itree.batches),
                  at::cuda::getCurrentCUDAStream());
"""
                )
            return (
                common
                + f"""
      const int32_t part_count = static_cast<int32_t>(itree.batches);
      const int32_t rows = static_cast<int32_t>(M);
      auto part = self.new_empty({{M * itree.batches}},
                                 self.options().dtype(at::kFloat));
      {launch_fn}(self_2d, part, out_1d, rows, part_count,
                  static_cast<int32_t>(N), at::cuda::getCurrentCUDAStream());
"""
            )
        if spec["path"] == "xcta":
            subrow = int(_DATA.XCTA_SUBROW)
            return f"""
      const int32_t part_count = static_cast<int32_t>(partials);
      const int32_t rows = static_cast<int32_t>(M * partials);
      auto sub = self.view({{rows, {subrow}}});
      auto part = self.new_empty({{rows}}, self.options().dtype(at::kFloat));
      auto out_1d = out.view({{M}});
      {launch_fn}(sub, part, out_1d, rows, part_count, static_cast<int32_t>(M), N,
                  at::cuda::getCurrentCUDAStream());
    """
        vec = 16 // _DATA.DTYPE_BYTES[spec["dtype"]]
        threads_per_row, _ = ROW_CLASSES[spec["row_class"]]
        return f"""
      auto self_2d = self.view({{M, N}});
      auto out_1d = out.view({{M}});
      const int32_t nchunks = static_cast<int32_t>(N / {vec});
      const int32_t nwaves = (nchunks + {threads_per_row - 1}) / {threads_per_row};
      {launch_fn}(self_2d, out_1d, nchunks, nwaves, static_cast<int32_t>(N),
                  at::cuda::getCurrentCUDAStream());
    """


class _IndexReduction:
    DISPATCH_KEY = "CUDA"
    KERNEL_MODULE = "aot_kernel.py"
    ARCHS = ARCHS

    def __init__(self, aten_op, trait, dim_style, out_specs, outputs):
        self.ATEN_OP = aten_op
        self._trait = trait
        self._dim_style = dim_style
        self._out_specs = out_specs
        self._outputs = outputs

    def _out_dtypes(self, dtype):
        return tuple(dtype if value == "input" else value for value in self._out_specs)

    def kernel_precompile_grid(self):
        return [
            {
                "path": "row",
                "trait": self._trait,
                "dtype": dtype,
                "out_dtypes": self._out_dtypes(dtype),
                "row_class": row_class,
                "order": "linear",
            }
            for dtype in DTYPES
            for row_class in ROW_CLASSES
        ]

    def covered_axes(self, self_t, dim=None, keepdim=False):
        import torch

        names = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }
        dtype_name = names.get(self_t.dtype)
        geometry = _row_geometry(self_t, dim, self._dim_style)
        row_class = None
        if geometry is not None and dtype_name is not None:
            n, m, reduce_all = geometry
            row_class = _DATA.row_class(dtype_name, n, m, reduce_all)
        eligible = (
            self_t.is_cuda
            and self_t.is_contiguous()
            and dtype_name is not None
            and self_t.storage_offset() * self_t.element_size() % 16 == 0
            and row_class is not None
        )
        return {
            "path": "row" if eligible else None,
            "trait": self._trait,
            "dtype": self_t.dtype,
            "row_class": row_class,
            "order": "linear",
        }

    def _output_checks(self, *, output):
        checks = []
        for name, out_spec in zip(self._outputs, self._out_specs):
            dtype = "st" if out_spec == "input" else _DTYPE_CPP[out_spec]
            access = f"{name}." if output else f"{name}->"
            check = (
                f"{access}scalar_type() != {dtype} || !{access}is_contiguous() || "
                f"reinterpret_cast<uintptr_t>({access}const_data_ptr()) % 16 != 0"
            )
            if output:
                checks.append(f"      if ({check}) return false;")
            else:
                checks.append(
                    f"      if ({name}.has_value() && ({check})) return false;"
                )
        return "\n".join(checks)

    def _front(self, *, output):
        dtype_accept = " || ".join(f"st == {_DTYPE_CPP[dtype]}" for dtype in DTYPES)
        return f"""
      const auto st = self.scalar_type();
      if (!({dtype_accept})) return false;
      if (self.dim() < 1 || !self.is_contiguous()) return false;
      if (reinterpret_cast<uintptr_t>(self.const_data_ptr()) % 16 != 0) return false;
{self._output_checks(output=output)}
      {_geometry_cpp(self._dim_style)}
      {_classifier_cpp()}
      if (!use_row) return false;
    """

    def cpp_covers(self):
        return f"{self._front(output=False)}\n      return true;"

    def cpp_dispatch_prelude(self):
        return self._front(output=True)

    def cpp_dispatch(self, spec):
        threads_per_row, threads_per_block = ROW_CLASSES[spec["row_class"]]
        return (
            f"st == {_DTYPE_CPP[spec['dtype']]} && "
            f"threads_per_row == {threads_per_row} && "
            f"threads_per_block == {threads_per_block}"
        )

    def cpp_launch(self, spec, launch_fn):
        vec = 16 // _DATA.DTYPE_BYTES[spec["dtype"]]
        threads_per_row, _ = ROW_CLASSES[spec["row_class"]]
        output_views = "\n".join(
            f"      auto {name}_1d = {name}.view({{M}});" for name in self._outputs
        )
        launch_outputs = ", ".join(f"{name}_1d" for name in self._outputs)
        return f"""
      auto self_2d = self.view({{M, N}});
{output_views}
      const int32_t nchunks = static_cast<int32_t>(N / {vec});
      const int32_t nwaves = (nchunks + {threads_per_row - 1}) / {threads_per_row};
      {launch_fn}(self_2d, {launch_outputs}, nchunks, nwaves,
                  static_cast<int32_t>(N), at::cuda::getCurrentCUDAStream());
    """


def declarations():
    return [
        *(_RowReduction(*row) for row in CORE_OPS),
        *(_IndexReduction(*row) for row in INDEX_OPS),
    ]
