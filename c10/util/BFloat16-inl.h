#pragma once

#include <c10/macros/Macros.h>

namespace c10 {

/// Constructors

inline C10_HOST_DEVICE BFloat16::BFloat16(float value) {
  uint32_t res;
#if defined(__CUDA_ARCH__)
  cudaMemcpy(&res, &value, sizeof(res), cudaMemcpyDeviceToHost);
#else
  std::memcpy(&res, &value, sizeof(res));
#endif
  val_ = res >> 16;
}

/// Implicit conversions
inline C10_HOST_DEVICE BFloat16::operator float() const {
  float res = 0;
  uint32_t tmp = val_;
  tmp <<= 16;
  std::memcpy(&res, &tmp, sizeof(tmp));
  return res;
}

} // namespace c10
