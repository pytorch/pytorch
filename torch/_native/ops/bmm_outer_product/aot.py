"""Native-AOT declaration for aten::bmm @ CUDA (outer-product, K == 1).

First triton-kind op. Precompile points carry BLOCK_M/BLOCK_N (which
compiled kernel) plus the (lo, hi] M-bucket the JIT wrapper's block-size
picker assigns them -- cpp_dispatch renders the bucket as a range
condition, mirrored by covered_axes' bucket booleans. Narrow starter
grid: fp32/bf16, M in (32,192], N >= 128; fp16, other buckets, small N,
and non-CUDA accelerators stay with the JIT override.
"""

ATEN_OP = "bmm"
DISPATCH_KEY = "CUDA"
KERNEL_MODULE = "aot_kernel.py"

_DTYPES = {"float32": "at::kFloat", "bfloat16": "at::kBFloat16"}
# (BLOCK_M, m_lo, m_hi): one compiled kernel per (dtype, BLOCK_M),
# serving M in (m_lo, m_hi]. Matches _pick_block_sizes in
# triton_kernels.py for these ranges (BLOCK_N = 128 there).
_M_BUCKETS = [(32, 32, 96), (64, 96, 192)]
_MIN_N = 128


def kernel_precompile_grid():
    # NB: coverage matches a call iff some point agrees on every field
    # covered_axes() returns -- so "outer" must appear here (pinned True)
    # or non-outer calls would falsely match on dtype alone. BLOCK_* and
    # m_lo/m_hi are export/dispatch-only (absent from covered_axes).
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
    covered_bucket = any(lo < m <= hi for _, lo, hi in _M_BUCKETS)
    # Mirror the prelude's specialization conditions (baked innermost
    # strides + 16B-aligned pointers) so covered calls don't silently
    # fall to stock aten when the C++ side declines. Alignment via
    # storage_offset, NOT data_ptr(): Python data_ptr() materializes
    # copy-on-write inputs and coverage runs on every call. Allocator
    # bases are >=256B aligned; the C++ prelude re-checks the pointer.
    specialized = (
        is_outer
        and self.stride(1) == 1
        and mat2.stride(2) == 1
        and (self.storage_offset() * self.element_size()) % 16 == 0
        and (mat2.storage_offset() * mat2.element_size()) % 16 == 0
    )
    return {
        "dtype": self.dtype,
        "outer": specialized and covered_bucket and mat2.shape[2] >= _MIN_N,
    }


def cpp_dispatch_prelude():
    dtype_reject = " && ".join(f"st != {t}" for t in _DTYPES.values())
    return f"""
      const auto st = self.scalar_type();
      if ({dtype_reject}) return false;
      if (self.size(2) != 1 || mat2.size(1) != 1) return false;
      if (self.numel() == 0 || mat2.numel() == 0) return false;
      if (self.size(0) > std::numeric_limits<int32_t>::max()) return false;
      // i32 stride parity with the JIT specialization (values checked;
      // out is dense from meta so its strides are bounded by numel).
      if (self.stride(0) > std::numeric_limits<int32_t>::max() ||
          mat2.stride(0) > std::numeric_limits<int32_t>::max() ||
          out.numel() > std::numeric_limits<int32_t>::max()) return false;
      const int64_t M = self.size(1);
      const int64_t N = mat2.size(2);
      if (N < {_MIN_N}) return false;
      // Specialization parity with the exported kernels: innermost
      // strides are baked to 1 and pointers hinted 16B-aligned.
      if (self.stride(1) != 1 || mat2.stride(2) != 1 || out.stride(2) != 1) return false;
      // const_data_ptr on inputs: data_ptr() would materialize COW.
      if (reinterpret_cast<std::uintptr_t>(self.const_data_ptr()) % 16 != 0 ||
          reinterpret_cast<std::uintptr_t>(mat2.const_data_ptr()) % 16 != 0 ||
          reinterpret_cast<std::uintptr_t>(out.data_ptr()) % 16 != 0) return false;
    """


def cpp_dispatch(spec):
    return (
        f"st == {_DTYPES[spec['dtype']]} && M > {spec['m_lo']} && M <= {spec['m_hi']}"
    )


def cpp_launch(spec, launch_fn):
    return f"""
      {launch_fn}(self, mat2, out,
                static_cast<int32_t>(self.size(0)), static_cast<int32_t>(M), static_cast<int32_t>(N),
                static_cast<int32_t>(self.stride(0)),
                static_cast<int32_t>(mat2.stride(0)),
                static_cast<int32_t>(out.stride(0)), static_cast<int32_t>(out.stride(1)),
                at::cuda::getCurrentCUDAStream());
    """
