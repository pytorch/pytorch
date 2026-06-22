#pragma once

#include <ATen/native/cuda/MemoryAccess.cuh>

namespace c10d::symmetric_memory {

// Cooperative 16-byte-vectorized copy of `nbytes` from `src` to `dst`.  Caller
// must ensure both pointers are 16-byte aligned and nbytes is a multiple of 16.
// `src` and `dst` are `__restrict__`: the two regions must not overlap, which
// lets the compiler pipeline loads ahead of stores.  Overlapping ranges are
// undefined behavior and are NOT diagnosed (the contract is unchecked), so only
// use this for distinct buffers -- both callers copy between separate input and
// output windows, never in place.
// `tid`/`stride` are this thread's rank and the cooperating thread count (e.g.
// threadIdx.x / blockDim.x for a CTA-wide copy).  Loads are batched kUnroll-deep
// before the stores so several loads stay in flight (memory-level parallelism),
// which is what hides latency when occupancy is low (few CTAs).
__device__ inline void copy_bytes_vec16_aligned(
    const char* __restrict__ src,
    char* __restrict__ dst,
    size_t nbytes,
    size_t tid,
    size_t stride) {
  const size_t n_vec = nbytes / 16;
  constexpr int kUnroll = 4;
  size_t vec_idx = tid;
  for (; vec_idx + static_cast<size_t>(kUnroll - 1) * stride < n_vec;
       vec_idx += static_cast<size_t>(kUnroll) * stride) {
    at::native::memory::Vec<16> chunk[kUnroll];
#pragma unroll 4
    for (int u = 0; u < kUnroll; ++u) {
      const size_t i = vec_idx + static_cast<size_t>(u) * stride;
      chunk[u] = at::native::memory::ld_vec<16>(src + i * 16);
    }
#pragma unroll 4
    for (int u = 0; u < kUnroll; ++u) {
      const size_t i = vec_idx + static_cast<size_t>(u) * stride;
      at::native::memory::st_vec<16>(dst + i * 16, chunk[u]);
    }
  }
  for (; vec_idx < n_vec; vec_idx += stride) {
    const char* src_ptr = src + vec_idx * 16;
    char* dst_ptr = dst + vec_idx * 16;
    auto v = at::native::memory::ld_vec<16>(src_ptr);
    at::native::memory::st_vec<16>(dst_ptr, v);
  }
}

} // namespace c10d::symmetric_memory
