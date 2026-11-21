#pragma once

#include <algorithm>
#include <cstdint>
#include <initializer_list>

// NOTE: no `namespace at::native` wrapper here. Every includer of this file
// lives in the dispatch chain, which CUDALoops.cuh includes from inside its
// own `namespace at::native` block, so the contents inherit that scope.
// Re-wrapping would create a nested `at` member (at::native::at::native)
// that shadows every `at::` qualified reference in the includer.

// Per-thread vector width in elements for same-dtype kernels (nocast
// family). 16 bytes is the widest useful access, so bf16 gives 8 (one
// 16-byte access). Note this deliberately does NOT copy
// launch_vectorized_kernel's extra "cap at 4 unless sm_90/sm_100" rule: that
// cap exists because upstream's own vec8 kernel body is compiled away
// outside sm_90/sm_100, which does not apply to the kernels below.
template <typename T>
constexpr int vector_access_width() {
  constexpr int by_size = 16 / sizeof(T);
  return by_size < 8 ? by_size : 8;
}

// Per-operand 16-byte vector width for cast kernels: each operand issues
// its own aligned_vector<T, per_op_vector_width<T>()> accesses (full 16
// bytes per access), and the caller takes max() across operands as the
// per-thread element count (lcm, powers of two).
template <typename T>
constexpr int per_op_vector_width() {
  return 16 / sizeof(T);
}

// aligned_vector<T, vt> can be lowered to several ISA loads/stores.  The
// dispatcher must gate every participating pointer and higher-dimension base
// stride by the width of one emitted ISA access, not by a different operand's
// element size or by the aggregate logical vector size.
template <typename T, int vt>
constexpr int64_t vector_isa_access_bytes() {
  constexpr int64_t logical_bytes = static_cast<int64_t>(sizeof(T)) * vt;
  return logical_bytes < 16 ? logical_bytes : 16;
}

static inline bool is_vector_access_aligned(
    const char* ptr,
    int64_t isa_access_bytes,
    std::initializer_list<int64_t> higher_dim_strides = {}) {
  if (isa_access_bytes <= 0 ||
      reinterpret_cast<uintptr_t>(ptr) % isa_access_bytes != 0) {
    return false;
  }
  for (const int64_t stride : higher_dim_strides) {
    if (stride % isa_access_bytes != 0) {
      return false;
    }
  }
  return true;
}

template <typename scalar_t, int vt>
static inline bool is_vector_access_aligned(
    const char* ptr,
    std::initializer_list<int64_t> higher_dim_strides = {}) {
  return is_vector_access_aligned(
      ptr, vector_isa_access_bytes<scalar_t, vt>(), higher_dim_strides);
}
