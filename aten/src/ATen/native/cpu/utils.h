#pragma once

#include <ATen/cpu/vec/vec.h>
#include <c10/util/llvmMathExtras.h>

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#endif

namespace at {
namespace native {

inline namespace CPU_CAPABILITY {

template <typename T>
inline T data_index_init(T offset) {
  return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T& x, const T& X, Args&&... args) {
  offset = data_index_init(offset, std::forward<Args>(args)...);
  x = offset % X;
  return offset / X;
}

inline bool data_index_step() {
  return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T& x, const T& X, Args&&... args) {
  if (data_index_step(std::forward<Args>(args)...)) {
    x = ((x + 1) == X) ? 0 : (x + 1);
    return x == 0;
  }
  return false;
}

} // namespace

namespace utils {

template <typename T>
T CeilLog2(const T& x) {
  if (x <= 2) {
    return 1;
  }
  // Last set bit is floor(log2(x)), floor + 1 is ceil
  // except when x is an exact powers of 2, so subtract 1 first
  return static_cast<T>(llvm::findLastSet(static_cast<uint64_t>(x) - 1)) + 1;
}

// matrix transpose:
//   src has shape of M by N, with leading dimension of ld_src
//   dst has shape of N by M, with leading dimension of ld_dst
template <typename T>
inline void transpose(int64_t M, int64_t N, const T* src, int64_t ld_src, T* dst, int64_t ld_dst) {
  for (int64_t j = 0; j < N; j++) {
    for (int64_t i = 0; i < M; i++) {
      dst[j * ld_dst + i] = src[i * ld_src + j];
    }
  }
}

#ifdef USE_FBGEMM
template <>
inline void transpose<float>(int64_t M, int64_t N, const float* src, int64_t ld_src, float* dst, int64_t ld_dst) {
  TORCH_CHECK(fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");
  fbgemm::transpose_simd<float>(M, N, src, ld_src, dst, ld_dst);
}
#endif

} // namespace utils

} // namespace native
} // namespace at
