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
WELFORD_OPS = _DATA.WELFORD_OPS
PLAIN_OPS = _DATA.PLAIN_OPS
DTYPE_OPS = _DATA.DTYPE_OPS
NORM_TRAITS = _DATA.NORM_TRAITS
DTYPES = _DATA.DTYPES
ROW_CLASSES = _DATA.ROW_CLASSES
ORDER_SENSITIVE_TRAITS = _DATA.ORDER_SENSITIVE_TRAITS


_DTYPE_CPP = {
    "bool": "at::kBool",
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
    if style == "full":
        n = self.numel()
        return (n, 1, True) if n else None
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


def _geometry_cpp(style, *, expose_first=False):
    if style == "full":
        first = "const int64_t first = 0;" if expose_first else ""
        return f"""
      const bool row_dims = self.dim() > 0;
      const bool reduce_all = true;
      const int64_t N = self.numel();
      {first}
    """

    if style == "int":
        first = "const int64_t first = self.dim() - 1;" if expose_first else ""
        return """
      bool row_dims = false;
      const bool reduce_all = false;
      int64_t N = 0;
      @FIRST@
      if (self.dim() > 0) {
        const int64_t wrapped = c10::maybe_wrap_dim(dim, self.dim());
        row_dims = wrapped == self.dim() - 1;
        if (row_dims) N = self.size(wrapped);
      }
    """.replace("@FIRST@", first)

    if style == "optional_int":
        first = (
            "const int64_t first = reduce_all ? 0 : self.dim() - 1;"
            if expose_first
            else ""
        )
        return """
      const bool reduce_all = !dim.has_value();
      bool row_dims = self.dim() > 0;
      int64_t N = 0;
      @FIRST@
      if (reduce_all) {
        N = self.numel();
      } else if (row_dims) {
        const int64_t wrapped = c10::maybe_wrap_dim(*dim, self.dim());
        row_dims = wrapped == self.dim() - 1;
        if (row_dims) N = self.size(wrapped);
      }
    """.replace("@FIRST@", first)

    dims = "dim" if style == "list" else "*dim"
    reduce_all = (
        "dim.empty()" if style == "list" else "!dim.has_value() || dim->empty()"
    )
    first_decl = "int64_t first = 0;" if expose_first else ""
    first_assign = (
        "first = self.dim() - k;"
        if expose_first
        else "const int64_t first = self.dim() - k;"
    )
    return f"""
      bool row_dims = self.dim() > 0;
      const bool reduce_all = {reduce_all};
      int64_t N = 0;
      {first_decl}
      if (reduce_all) {{
        N = self.numel();
      }} else {{
        const auto dims = {dims};
        const int64_t k = static_cast<int64_t>(dims.size());
        row_dims = row_dims && k > 0 && k <= self.dim();
        {first_assign}
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
        self.ATEN_ALIASES = _DATA.AOT_ALIASES.get(aten_op, ())
        self.KERNEL_SOURCE = _DATA.KERNEL_SOURCES.get(aten_op, aten_op)
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

    @staticmethod
    def cpp_helpers():
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
        self.ATEN_ALIASES = _DATA.AOT_ALIASES.get(aten_op, ())
        self.KERNEL_SOURCE = _DATA.KERNEL_SOURCES.get(aten_op, aten_op)
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


class _ExtendedReduction:
    DISPATCH_KEY = "CUDA"
    KERNEL_MODULE = "aot_kernel.py"
    ARCHS = ARCHS

    def __init__(
        self, aten_op, traits, dim_style, out_specs, outputs, has_keepdim=True
    ):
        self.ATEN_OP = aten_op
        self.ATEN_ALIASES = _DATA.AOT_ALIASES.get(aten_op, ())
        self.KERNEL_SOURCE = _DATA.KERNEL_SOURCES.get(aten_op, aten_op)
        self.STRUCTURED = bool(outputs)
        self._traits = tuple(traits)
        self._dim_style = dim_style
        self._out_specs = tuple(out_specs)
        self._outputs = tuple(outputs)
        self._has_dtype = "dtype" in self._out_specs
        self._has_keepdim = has_keepdim

    def _out_dtypes(self, dtype):
        if self._has_dtype:
            return ((out_dtype,) for out_dtype in DTYPES)
        return (
            tuple(dtype if value == "input" else value for value in self._out_specs),
        )

    @staticmethod
    def _point(path, trait, dtype, out_dtypes, **kwargs):
        point = {
            "path": path,
            "trait": trait,
            "dtype": dtype,
            "order": "linear",
            **kwargs,
        }
        if "dtype" in out_dtypes:
            raise AssertionError("output dtype must be resolved before grid expansion")
        if len(out_dtypes) == 1:
            point["out_dtype"] = out_dtypes[0]
        else:
            point["out_dtypes"] = out_dtypes
        return point

    def kernel_precompile_grid(self):
        points = []
        for trait in self._traits:
            for dtype in DTYPES:
                for out_dtypes in self._out_dtypes(dtype):
                    for row_class in ROW_CLASSES:
                        points.append(
                            self._point(
                                "row",
                                trait,
                                dtype,
                                out_dtypes,
                                row_class=row_class,
                            )
                        )
                    points.append(self._point("xcta", trait, dtype, out_dtypes))
                    if trait in ORDER_SENSITIVE_TRAITS:
                        for path in ("itree_multirow", "itree_looped"):
                            for loads in _DATA.INNER_TREE_LOADS:
                                point = self._point(
                                    path, trait, dtype, out_dtypes, loads=loads
                                )
                                point["order"] = "inner_tree"
                                points.append(point)
                        point = self._point(
                            "itree_split",
                            trait,
                            dtype,
                            out_dtypes,
                            loads=2 if _DATA.DTYPE_BYTES[dtype] == 2 else 4,
                        )
                        point["order"] = "inner_tree"
                        points.append(point)
        return points

    def _covered_axes(self, self_t, dim, trait, out_dtype=None):
        import torch

        names = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }
        dtype_name = names.get(self_t.dtype)
        out_dtype_name = next(
            (name for value, name in names.items() if value == out_dtype), None
        )
        geometry = _row_geometry(self_t, dim, self._dim_style)
        ordered = trait in ORDER_SENSITIVE_TRAITS and any(
            os.environ.get(name, "") not in ("", "0") for name in _INNER_TREE_ENVS
        )
        row_class = None
        loads = None
        path = None
        if geometry is not None and dtype_name is not None and trait is not None:
            n, m, reduce_all = geometry
            row_class = _DATA.row_class(dtype_name, n, m, reduce_all)
            if row_class is not None:
                path = "row"
            elif _DATA.xcta_ok(dtype_name, n, m):
                path = "xcta"
            if ordered and path is not None:
                inner = _DATA.inner_tree_spec(dtype_name, n, m)
                if inner is None:
                    path = None
                else:
                    path, loads = inner
                    row_class = None
        eligible = (
            self_t.is_cuda
            and self_t.is_contiguous()
            and dtype_name is not None
            and (not self._has_dtype or out_dtype_name is not None)
            and self_t.storage_offset() * self_t.element_size() % 16 == 0
            and path is not None
        )
        axes = {
            "path": path if eligible else None,
            "trait": trait,
            "dtype": self_t.dtype,
            "row_class": row_class,
            "loads": loads,
            "order": "inner_tree" if ordered else "linear",
        }
        if self._has_dtype:
            axes["out_dtype"] = out_dtype
        return axes

    def cpp_helpers(self):
        return _RowReduction.cpp_helpers()

    def _output_checks(self, *, output):
        checks = []
        for name, out_spec in zip(self._outputs, self._out_specs):
            dtype = (
                "st"
                if out_spec == "input"
                else "out_st"
                if out_spec == "dtype"
                else _DTYPE_CPP[out_spec]
            )
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

    def _variant_cpp(self):
        return ""

    def _variant_dispatch(self, spec):
        return "true"

    def _front(self, *, output):
        dtype_accept = " || ".join(f"st == {_DTYPE_CPP[dtype]}" for dtype in DTYPES)
        out_dtype = ""
        if self._has_dtype:
            out_accept = " || ".join(
                f"out_st == {_DTYPE_CPP[dtype]}" for dtype in DTYPES
            )
            if output and self.STRUCTURED:
                out_dtype = f"""
      const auto out_st = {self._outputs[0]}.scalar_type();
      if (!({out_accept})) return false;
      if (dtype.has_value() && *dtype != out_st) return false;
"""
            else:
                out_dtype = f"""
      const auto out_st = dtype.value_or(st);
      if (!({out_accept})) return false;
"""
        inner_tree = (
            '_naot_reduction_env_enabled("PYTORCH_NATIVE_INNER_TREE") || '
            '_naot_reduction_env_enabled("PYTORCH_SUM_INNER_TREE")'
            if any(trait in ORDER_SENSITIVE_TRAITS for trait in self._traits)
            else "false"
        )
        output_shape = ""
        if not self.STRUCTURED:
            keepdim = "keepdim" if self._has_keepdim else "false"
            output_shape = f"""
      std::vector<int64_t> output_shape;
      for (int64_t axis = 0; axis < first; ++axis) {{
        output_shape.push_back(self.size(axis));
      }}
      if ({keepdim}) {{
        for (int64_t axis = first; axis < self.dim(); ++axis) {{
          output_shape.push_back(1);
        }}
      }}
"""
        return f"""
      const auto st = self.scalar_type();
      if (!({dtype_accept})) return false;
      if (self.dim() < 1 || !self.is_contiguous()) return false;
      if (reinterpret_cast<uintptr_t>(self.const_data_ptr()) % 16 != 0) return false;
{out_dtype}{self._output_checks(output=output)}
{self._variant_cpp()}
      {_geometry_cpp(self._dim_style, expose_first=not self.STRUCTURED)}
      {_classifier_cpp()}
      const bool inner_tree = {inner_tree};
      const auto itree = inner_tree
          ? _naot_inner_tree_plan(N, M, self.element_size())
          : _NaotInnerTreePlan{{0, 0, 0, 0}};
      if (inner_tree && itree.shape == 0) return false;
{output_shape}
    """

    def cpp_covers(self):
        return f"{self._front(output=False)}\n      return true;"

    def cpp_dispatch_prelude(self):
        return self._front(output=True)

    def cpp_dispatch(self, spec):
        common = f"st == {_DTYPE_CPP[spec['dtype']]} && {self._variant_dispatch(spec)}"
        if self._has_dtype:
            common += f" && out_st == {_DTYPE_CPP[spec['out_dtype']]}"
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

    def _spec_out_dtypes(self, spec):
        return (
            tuple(spec["out_dtypes"]) if "out_dtypes" in spec else (spec["out_dtype"],)
        )

    def _launch_tensors(self, spec):
        if self.STRUCTURED:
            return self._outputs, ""
        names = tuple(f"aot_out{i}" for i in range(len(self._out_specs)))
        allocations = []
        for name, dtype in zip(names, self._spec_out_dtypes(spec)):
            allocations.append(
                f"      auto {name} = at::empty(output_shape, "
                f"self.options().dtype({_DTYPE_CPP[dtype]}));"
            )
        return names, "\n".join(allocations)

    def _result_assignment(self, names):
        if self.STRUCTURED:
            return ""
        value = names[0] if len(names) == 1 else f"std::make_tuple({', '.join(names)})"
        return f"\n      aot_result = {value};"

    def cpp_launch(self, spec, launch_fn):
        outputs, allocations = self._launch_tensors(spec)
        output_views = "\n".join(
            f"      auto {name}_1d = {name}.view({{M}});" for name in outputs
        )
        output_args = ", ".join(f"{name}_1d" for name in outputs)
        assignment = self._result_assignment(outputs)
        common = f"""
{allocations}
      auto self_2d = self.view({{M, N}});
{output_views}
"""
        if spec["order"] == "inner_tree":
            if spec["path"] == "itree_multirow":
                return (
                    common
                    + f"""
      {launch_fn}(self_2d, {output_args}, static_cast<int32_t>(N),
                  static_cast<int32_t>(M), at::cuda::getCurrentCUDAStream());{assignment}
"""
                )
            if spec["path"] == "itree_looped":
                return (
                    common
                    + f"""
      {launch_fn}(self_2d, {output_args}, static_cast<int32_t>(N),
                  static_cast<int32_t>(M), static_cast<int32_t>(itree.warps),
                  static_cast<int32_t>(itree.batches),
                  at::cuda::getCurrentCUDAStream());{assignment}
"""
                )
            part_names = tuple(
                f"part{i}" for i in range(_DATA.TRAIT_NFIELDS[spec["trait"]])
            )
            parts = "\n".join(
                f"      auto {name} = self.new_empty({{M * itree.batches}}, "
                "self.options().dtype(at::kFloat));"
                for name in part_names
            )
            args = ", ".join((*part_names, *output_args.split(", ")))
            return (
                common
                + f"""
      const int32_t part_count = static_cast<int32_t>(itree.batches);
      const int32_t rows = static_cast<int32_t>(M);
{parts}
      {launch_fn}(self_2d, {args}, rows, part_count, static_cast<int32_t>(N),
                  at::cuda::getCurrentCUDAStream());{assignment}
"""
            )
        if spec["path"] == "xcta":
            subrow = int(_DATA.XCTA_SUBROW)
            part_names = tuple(
                f"part{i}" for i in range(_DATA.TRAIT_NFIELDS[spec["trait"]])
            )
            parts = "\n".join(
                f"      auto {name} = self.new_empty({{rows}}, "
                "self.options().dtype(at::kFloat));"
                for name in part_names
            )
            args = ", ".join((*part_names, *output_args.split(", ")))
            return f"""
{allocations}
      const int32_t part_count = static_cast<int32_t>(partials);
      const int32_t rows = static_cast<int32_t>(M * partials);
      auto sub = self.view({{rows, {subrow}}});
{parts}
{output_views}
      {launch_fn}(sub, {args}, rows, part_count, static_cast<int32_t>(M), N,
                  at::cuda::getCurrentCUDAStream());{assignment}
"""
        vec = 16 // _DATA.DTYPE_BYTES[spec["dtype"]]
        threads_per_row, _ = ROW_CLASSES[spec["row_class"]]
        return (
            common
            + f"""
      const int32_t nchunks = static_cast<int32_t>(N / {vec});
      const int32_t nwaves = (nchunks + {threads_per_row - 1}) / {threads_per_row};
      {launch_fn}(self_2d, {output_args}, nchunks, nwaves,
                  static_cast<int32_t>(N), at::cuda::getCurrentCUDAStream());{assignment}
"""
        )


class _WelfordReduction(_ExtendedReduction):
    def __init__(self, aten_op, trait, out_specs):
        super().__init__(
            aten_op,
            (f"{trait}0", f"{trait}1"),
            "optional_list",
            out_specs,
            (),
        )

    @staticmethod
    def _trait(trait, correction):
        try:
            value = 1.0 if correction is None else float(correction)
        except (TypeError, ValueError):
            return None
        return f"{trait}{int(value)}" if value in (0.0, 1.0) else None

    def covered_axes(self, self_t, dim=None, *, correction=None, keepdim=False):
        return self._covered_axes(
            self_t, dim, self._trait(self._traits[0][:-1], correction)
        )

    def _variant_cpp(self):
        return """
      const double aot_variant =
          correction.has_value() ? correction->toDouble() : 1.0;
      if (aot_variant != 0.0 && aot_variant != 1.0) return false;
"""

    def _variant_dispatch(self, spec):
        return f"aot_variant == {spec['trait'][-1]}.0"


class _NormReduction(_ExtendedReduction):
    def __init__(self):
        super().__init__(
            "linalg_vector_norm",
            NORM_TRAITS,
            "optional_list",
            ("dtype",),
            ("out",),
        )

    @staticmethod
    def _trait(ord_value):
        try:
            value = float(ord_value)
        except (TypeError, ValueError):
            return None
        return {
            1.0: "vnorm1",
            2.0: "vnorm2",
            float("inf"): "vnorminf",
            float("-inf"): "vnormninf",
        }.get(value)

    def covered_axes(self, self_t, ord=2, dim=None, keepdim=False, *, dtype=None):
        return self._covered_axes(
            self_t, dim, self._trait(ord), self_t.dtype if dtype is None else dtype
        )

    def _variant_cpp(self):
        return """
      const double aot_variant = ord.toDouble();
      const double aot_inf = std::numeric_limits<double>::infinity();
      if (aot_variant != 1.0 && aot_variant != 2.0 &&
          aot_variant != aot_inf && aot_variant != -aot_inf) return false;
"""

    def _variant_dispatch(self, spec):
        value = {
            "vnorm1": "1.0",
            "vnorm2": "2.0",
            "vnorminf": "aot_inf",
            "vnormninf": "-aot_inf",
        }[spec["trait"]]
        return f"aot_variant == {value}"


class _PlainReduction(_ExtendedReduction):
    def __init__(self, aten_op, trait, dim_style, out_specs, outputs, has_keepdim):
        super().__init__(
            aten_op,
            (trait,),
            dim_style,
            out_specs,
            outputs,
            has_keepdim=has_keepdim,
        )

    def covered_axes(self, self_t, dim=None, keepdim=False):
        if self.ATEN_OP in ("all.dims", "any.dims") and dim is not None and not dim:
            return self._covered_axes(self_t, dim, None)
        return self._covered_axes(self_t, dim, self._traits[0])

    def _variant_cpp(self):
        if self.ATEN_OP in ("all.dims", "any.dims"):
            return """
      if (dim.has_value() && dim->empty()) return false;
"""
        return ""


class _DtypeReduction(_ExtendedReduction):
    def __init__(self, aten_op, trait, dim_style, has_keepdim):
        super().__init__(
            aten_op,
            (trait,),
            dim_style,
            ("dtype",),
            (),
            has_keepdim=has_keepdim,
        )

    def covered_axes(self, self_t, dim=None, keepdim=False, *, dtype=None):
        return self._covered_axes(
            self_t, dim, self._traits[0], self_t.dtype if dtype is None else dtype
        )


def declarations():
    return [
        *(_RowReduction(*row) for row in CORE_OPS),
        *(_IndexReduction(*row) for row in INDEX_OPS),
        *(_WelfordReduction(*row) for row in WELFORD_OPS),
        _NormReduction(),
        *(_PlainReduction(*row) for row in PLAIN_OPS),
        *(_DtypeReduction(*row) for row in DTYPE_OPS),
    ]
