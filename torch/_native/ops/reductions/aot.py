"""Native-AOT declarations for the reduction family @ CUDA (K1 row path).

A FAMILY module: declarations() returns one declaration object per
covered op (sum.dim_IntList, mean.dim, amax, amin, prod.dim_int), all
built by one factory over a shared axis table. Coverage is the K1
one-shot row-reduce regime only: 2D-collapsible contiguous input,
reduction over exactly the last axis, N on a bucket, fp32/bf16,
default out dtype. Reduce-all, column, index traits, off-bucket N and
everything else stay JIT (PAIN_POINTS P5/P7 record why).

Module scope must import with stdlib alone (torchgen loads this
pre-build); torch is imported lazily inside covered_axes bodies.
"""

_DTYPES = {"float32": "at::kFloat", "bfloat16": "at::kBFloat16"}
# N buckets: >= 1024 keeps clear of the small-N tpr crash (P12); powers
# of two are 16B-aligned rows for every grid dtype.
_NS = [1024, 2048, 4096, 8192, 16384, 32768]

# op axis: (ATEN_OP, trait, has dtype= arg in schema, dim arg style)
#   dim styles: "opt_list" (sum/mean: int[1]? dim), "list" (amax/amin:
#   int[1] dim=[]), "int" (prod: int dim).
_OPS = [
    ("sum.dim_IntList", "sum", True, "opt_list"),
    ("mean.dim", "mean", True, "opt_list"),
    ("amax", "amax", False, "list"),
    ("amin", "amin", False, "list"),
    ("prod.dim_int", "prod", True, "int"),
]


def _covers_last_dim(self, dim, dim_style):
    # True iff dim reduces exactly the trailing axis of a >=1D tensor
    # whose leading axes collapse (contiguity checked separately).
    ndim = self.dim()
    if ndim < 1:
        return False
    if dim_style == "int":
        d = dim
    else:
        if dim is None:
            return False  # reduce-all -> xcta (JIT)
        dims = [dim] if isinstance(dim, int) else list(dim)
        if len(dims) != 1:
            return False
        d = dims[0]
    if not isinstance(d, int) or not (-ndim <= d < ndim):
        return False
    return d % ndim == ndim - 1


class _RowReduceDecl:
    DISPATCH_KEY = "CUDA"
    KERNEL_MODULE = "aot_kernel.py"

    def __init__(self, aten_op, trait, has_dtype_arg, dim_style):
        self.ATEN_OP = aten_op
        self._trait = trait
        self._has_dtype_arg = has_dtype_arg
        self._dim_style = dim_style

    def kernel_precompile_grid(self):
        return [{"trait": self._trait, "dtype": list(_DTYPES), "N": _NS}]

    def covered_axes(self, self_t, dim=None, keepdim=False, *, dtype=None):
        # Alignment via storage_offset, NOT data_ptr(): data_ptr()
        # materializes copy-on-write inputs, and coverage runs on every
        # call (PAIN_POINTS P15). Allocator bases are >=256B aligned, so
        # offset alignment implies pointer alignment for framework
        # tensors; exotic bases are re-checked by the C++ prelude.
        aligned = (self_t.storage_offset() * self_t.element_size()) % 16 == 0
        covered = (
            self_t.is_contiguous()
            and _covers_last_dim(self_t, dim, self._dim_style)
            and (dtype is None or dtype == self_t.dtype)
            and aligned
        )
        n = self_t.shape[-1] if self_t.dim() >= 1 else 0
        return {
            "trait": self._trait if covered else None,
            "dtype": self_t.dtype,
            "N": n,
        }

    def cpp_covers(self):
        # C++ port of covered_axes + grid matching (registered as
        # torch.ops._native_aot.covers_<decl_id>; ~1.3us vs ~3.8us for
        # the Python path). Same covered set: contiguous >=1D, reduce
        # over exactly the last axis, on-grid dtype, default out dtype,
        # 16B-aligned base, N on a bucket. The signature is the
        # FUNCTIONAL schema (dim arrives raw -- wrap before comparing,
        # PAIN_POINTS P14) plus a trailing optional out.
        dtype_reject = " && ".join(f"st != {t}" for t in _DTYPES.values())
        n_accept = " || ".join(f"N == {n}" for n in _NS)
        if self._dim_style == "int":
            last_dim = "const bool last = (c10::maybe_wrap_dim(dim, self.dim()) == self.dim() - 1);"
        elif self._dim_style == "list":
            last_dim = "const bool last = (dim.size() == 1 && c10::maybe_wrap_dim(dim[0], self.dim()) == self.dim() - 1);"
        else:
            last_dim = "const bool last = (dim.has_value() && dim->size() == 1 && c10::maybe_wrap_dim((*dim)[0], self.dim()) == self.dim() - 1);"
        dtype_arg = (
            "if (dtype.has_value() && *dtype != st) return false;"
            if self._has_dtype_arg
            else ""
        )
        return f"""
      const auto st = self.scalar_type();
      if ({dtype_reject}) return false;
      {dtype_arg}
      if (self.dim() < 1 || !self.is_contiguous()) return false;
      if (reinterpret_cast<uintptr_t>(self.const_data_ptr()) % 16 != 0) return false;
      {last_dim}
      if (!last) return false;
      const int64_t N = self.size(-1);
      if (N == 0 || self.numel() == 0) return false;
      return {n_accept};
    """

    # ---- C++ side ----

    def cpp_dispatch_prelude(self):
        dtype_reject = " && ".join(f"st != {t}" for t in _DTYPES.values())
        # Reduction dims arrive at the structured impl RAW (no
        # maybe_wrap_dim precompute, unlike e.g. index_add's dim), so
        # the negative spelling (dim=-1) must be wrapped here or it
        # silently declines to stock aten while covered_axes says
        # covered (PAIN_POINTS P14).
        if self._dim_style == "int":
            last_dim = (
                "const bool last = "
                "(c10::maybe_wrap_dim(dim, self.dim()) == self.dim() - 1);"
            )
        elif self._dim_style == "list":
            last_dim = (
                "const bool last = (dim.size() == 1 && "
                "c10::maybe_wrap_dim(dim[0], self.dim()) == self.dim() - 1);"
            )
        else:
            # sum/mean: OptionalIntArrayRef; empty/absent = reduce-all.
            last_dim = (
                "const bool last = (dim.has_value() && dim->size() == 1 && "
                "c10::maybe_wrap_dim((*dim)[0], self.dim()) == self.dim() - 1);"
            )
        dtype_arg = (
            "if (dtype.has_value() && *dtype != self.scalar_type()) return false;"
            if self._has_dtype_arg
            else ""
        )
        return f"""
      const auto st = self.scalar_type();
      if ({dtype_reject}) return false;
      {dtype_arg}
      if (self.dim() < 1 || !self.is_contiguous()) return false;
      if (reinterpret_cast<uintptr_t>(self.const_data_ptr()) % 16 != 0) return false;
      {last_dim}
      if (!last) return false;
      const int64_t N = self.size(-1);
      // N == 0 must be rejected BEFORE the division: numel()/0 is a
      // C++ integer division by zero (SIGFPE, found by the OpInfo
      // test_out sweep via _refs.linalg.norm's empty samples).
      if (N == 0) return false;
      const int64_t M = self.numel() / N;
      if (M == 0) return false;
    """

    def cpp_dispatch(self, spec):
        return f"st == {_DTYPES[spec['dtype']]} && N == {spec['N']}"

    def cpp_launch(self, spec, launch_fn):
        if spec["dtype"] == "float32":
            # Kernel writes fp32 = aten's out dtype: write out directly.
            return f"""
      auto self_2d = self.view({{M, N}});
      auto out_1d = out.view({{M}});
      {launch_fn}(self_2d, out_1d, at::cuda::getCurrentCUDAStream());
    """
        # bf16 in -> fp32 kernel out -> cast into aten's bf16 out (the
        # same .to() the JIT impl pays).
        return f"""
      auto self_2d = self.view({{M, N}});
      auto acc = at::empty({{M}}, self.options().dtype(at::kFloat));
      {launch_fn}(self_2d, acc, at::cuda::getCurrentCUDAStream());
      out.view({{M}}).copy_(acc);
    """


def declarations():
    return [_RowReduceDecl(*row) for row in _OPS]
