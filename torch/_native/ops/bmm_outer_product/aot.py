"""Native-AOT declaration for aten::bmm @ CUDA (outer-product, K == 1), the first
triton-kind op. A point is (dtype, BLOCK_M) plus the M range it serves; fp16, N < 128 and
non-CUDA stay with the JIT override."""

ATEN_OP = "bmm"
DISPATCH_KEY = "CUDA"
KERNEL_MODULE = "aot_kernel.py"
# Stated, or the default claims every KNOWN_ARCHES. Both spellings, since either can
# appear in TORCH_CUDA_ARCH_LIST and export matches them exactly.
ARCHS = ("sm_90", "sm_90a", "sm_100", "sm_100a")

_DTYPES = {"float32": "at::kFloat", "bfloat16": "at::kBFloat16"}
# (BLOCK_M, m_lo, m_hi], m_hi None being unbounded: the kernel masks the M tail, so the
# edges are a perf choice, measured within noise of _pick_block_sizes' on an H100. Two
# points rather than one per M: a fixed BLOCK_M=64 halves occupancy at small M. The
# worst measured case is M=8 at 0.94x of the JIT's next_power_of_2 choice, taken over
# more points because coverage moves these calls off the JIT route entirely, so an
# extra bucket edge buys nothing a compile does not cost back.
_M_BUCKETS = [(32, 0, 96), (64, 96, None)]
_MIN_N = 128
_INT32_MAX = 2**31 - 1


def kernel_precompile_grid():
    # "outer" must appear, or a non-outer call would match on dtype alone.
    return [
        {
            "dtype": list(_DTYPES),
            "outer": True,
            "BLOCK_M": bm,
            "BLOCK_N": 128,
            "m_lo": lo,
            "m_hi": hi,
        }
        for bm, lo, hi in _M_BUCKETS
    ]


def covered_axes(self, mat2):
    is_outer = (
        self.dim() == 3
        and self.shape[2] == 1
        and mat2.shape[1] == 1
        and self.numel() > 0
        and mat2.numel() > 0
    )
    m = self.shape[1] if is_outer else 0
    covered_bucket = any(lo < m and (hi is None or m <= hi) for _, lo, hi in _M_BUCKETS)
    # Mirrors the prelude; alignment via storage_offset, since data_ptr() would
    # materialize COW on every call.
    specialized = (
        is_outer
        and self.stride(1) == 1
        and mat2.stride(2) == 1
        and (self.storage_offset() * self.element_size()) % 16 == 0
        and (mat2.storage_offset() * mat2.element_size()) % 16 == 0
    )
    # Also the prelude's: the ABI narrows these to int32_t. Coverage wider than the
    # stub's acceptance costs the call its JIT route as well, so the bounds belong on
    # both sides. A functional out is contiguous (B, M, N), so the one product bounds
    # its numel and both its strides; an out= call has no JIT route to lose and is
    # bounded by the prelude itself.
    int32_ok = (
        is_outer
        and self.shape[0] <= _INT32_MAX
        and self.stride(0) <= _INT32_MAX
        and mat2.stride(0) <= _INT32_MAX
        and self.numel() * mat2.shape[2] <= _INT32_MAX
    )
    return {
        "dtype": self.dtype,
        "outer": (
            specialized and covered_bucket and int32_ok and mat2.shape[2] >= _MIN_N
        ),
    }


def cpp_covers():
    # C++ port of covered_axes, registered as torch.ops._native_aot.covers_bmm, which
    # is also what gets the injected arch gate: without one, coverage would claim a
    # capability this declaration never exported, and the call would lose the JIT
    # route to a stub that then declines. Layout of out is not part of coverage --
    # only the functional overload has a JIT override to lose.
    dtype_accept = " || ".join(f"st == {t}" for t in _DTYPES.values())
    buckets = " || ".join(
        f"(M > {lo}" + (f" && M <= {hi})" if hi is not None else ")")
        for _, lo, hi in _M_BUCKETS
    )
    return f"""
      const auto st = self.scalar_type();
      if (!({dtype_accept})) return false;
      if (self.dim() != 3 || mat2.dim() != 3) return false;
      if (self.size(2) != 1 || mat2.size(1) != 1) return false;
      if (self.numel() == 0 || mat2.numel() == 0) return false;
      const int64_t M = self.size(1);
      const int64_t N = mat2.size(2);
      if (N < {_MIN_N}) return false;
      if (self.stride(1) != 1 || mat2.stride(2) != 1) return false;
      // const_data_ptr: data_ptr() would materialize COW.
      if (reinterpret_cast<std::uintptr_t>(self.const_data_ptr()) % 16 != 0 ||
          reinterpret_cast<std::uintptr_t>(mat2.const_data_ptr()) % 16 != 0) return false;
      if (self.size(0) > std::numeric_limits<int32_t>::max()) return false;
      if (self.stride(0) > std::numeric_limits<int32_t>::max() ||
          mat2.stride(0) > std::numeric_limits<int32_t>::max()) return false;
      // A functional out is contiguous (B, M, N), so this bounds its numel and both
      // of the strides the ABI narrows.
      if (self.numel() * N > std::numeric_limits<int32_t>::max()) return false;
      return {buckets};
    """


def cpp_dispatch_prelude():
    dtype_reject = " && ".join(f"st != {t}" for t in _DTYPES.values())
    return f"""
      const auto st = self.scalar_type();
      if ({dtype_reject}) return false;
      // aten's own dtype-equality check lives in the meta function, which this path
      // never runs: without this, a f32 kernel would write an out that bmm allocated
      // from mat2's bf16 options.
      if (mat2.scalar_type() != st || out.scalar_type() != st) return false;
      if (self.size(2) != 1 || mat2.size(1) != 1) return false;
      if (self.numel() == 0 || mat2.numel() == 0) return false;
      if (self.size(0) > std::numeric_limits<int32_t>::max()) return false;
      // i32 stride parity with the JIT specialization. out's own strides need their
      // own test: a view into a large base carries a stride past INT32_MAX while its
      // numel stays small.
      if (self.stride(0) > std::numeric_limits<int32_t>::max() ||
          mat2.stride(0) > std::numeric_limits<int32_t>::max() ||
          out.stride(0) > std::numeric_limits<int32_t>::max() ||
          out.stride(1) > std::numeric_limits<int32_t>::max() ||
          out.numel() > std::numeric_limits<int32_t>::max()) return false;
      const int64_t M = self.size(1);
      const int64_t N = mat2.size(2);
      if (N < {_MIN_N}) return false;
      // Parity with the exported kernels: innermost strides baked to 1, 16B alignment.
      if (self.stride(1) != 1 || mat2.stride(2) != 1 || out.stride(2) != 1) return false;
      // const_data_ptr: data_ptr() would materialize COW.
      if (reinterpret_cast<std::uintptr_t>(self.const_data_ptr()) % 16 != 0 ||
          reinterpret_cast<std::uintptr_t>(mat2.const_data_ptr()) % 16 != 0 ||
          reinterpret_cast<std::uintptr_t>(out.data_ptr()) % 16 != 0) return false;
    """


def cpp_dispatch(spec):
    # M needs no upper test: the prelude's out.numel() <= INT32_MAX bounds it.
    bounds = f"M > {spec['m_lo']}"
    if spec["m_hi"] is not None:
        bounds += f" && M <= {spec['m_hi']}"
    return f"st == {_DTYPES[spec['dtype']]} && {bounds}"


def cpp_launch(spec, launch_fn):
    return f"""
      {launch_fn}(self, mat2, out,
                static_cast<int32_t>(self.size(0)), static_cast<int32_t>(M), static_cast<int32_t>(N),
                static_cast<int32_t>(self.stride(0)),
                static_cast<int32_t>(mat2.stride(0)),
                static_cast<int32_t>(out.stride(0)), static_cast<int32_t>(out.stride(1)),
                at::cuda::getCurrentCUDAStream());
    """
