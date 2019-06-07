#pragma once

#include <cstring>
#include <limits>
#include <c10/macros/Macros.h>

namespace c10 {

/// Constructors

inline C10_HOST_DEVICE BFloat16::BFloat16(float value) {
  val_ = detail::bf16_from_f32(value);
}

/// Implicit conversions

inline C10_HOST_DEVICE BFloat16::operator float() const {
  return detail::f32_from_bf16(val_);
}

// CUDA intrinsics

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
inline __device__ BFloat16 __ldg(const BFloat16* ptr) {
    return __ldg(reinterpret_cast<const BFloat16*>(ptr));
}
#endif

} // namespace c10
