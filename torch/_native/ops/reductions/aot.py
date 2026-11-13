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
DTYPES = _DATA.DTYPES
ROW_CLASSES = _DATA.ROW_CLASSES


_DTYPE_CPP = {
    "float16": "at::kHalf",
    "bfloat16": "at::kBFloat16",
    "float32": "at::kFloat",
}
_INNER_TREE_ENV = "PYTORCH_NATIVE_INNER_TREE"
_SUM_INNER_TREE_ENV = "PYTORCH_SUM_INNER_TREE"


def _env_enabled(name) -> bool:
    # "" and "0" are off; any other value, including "strict", requests the order.
    return os.environ.get(name, "") not in ("", "0")


def _out_dtypes(dtype, has_dtype):
    # Like the JIT overrides, dtype= may only keep the input dtype or widen 16-bit
    # inputs to FP32; ATen casts the input before reducing for any other request.
    if has_dtype and dtype != "float32":
        return (dtype, "float32")
    return (dtype,)


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
        from torch._native.utils.tensor import const_data_ptr

        names = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }
        dtype_name = names.get(self_t.dtype)
        out_dtype = self_t.dtype if not self._has_dtype or dtype is None else dtype
        out_dtype_name = names.get(out_dtype)
        geometry = _row_geometry(self_t, dim, self._dim_style)
        ordered = any(_env_enabled(name) for name in self._order_envs)
        row_class = None
        path = None
        if geometry is not None and dtype_name is not None and not ordered:
            n, m, reduce_all = geometry
            row_class = _DATA.row_class(dtype_name, n, m, reduce_all)
            if row_class is not None:
                path = "row"
            elif _DATA.xcta_ok(dtype_name, n, m):
                path = "xcta"
        eligible = (
            self_t.is_cuda
            and self_t.is_contiguous()
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
        return """
inline bool _naot_reduction_env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\\0' &&
      !(value[0] == '0' && value[1] == '\\0');
}
"""

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
            out_st = "dtype.value_or(st)" if self._has_dtype else "st"
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
      if (self.dim() < 1 || !self.is_contiguous()) return false;
      if (reinterpret_cast<uintptr_t>(self.const_data_ptr()) % 16 != 0) return false;
      {out}
      {_geometry_cpp(self._dim_style)}
      {_classifier_cpp()}
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


def declarations():
    return [_RowReduction(*row) for row in CORE_OPS]
