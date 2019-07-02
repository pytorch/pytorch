#pragma once

// Defines the bloat16 type (brain floating-point). This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.

#include <c10/macros/Macros.h>
#include <cmath>
#include <cstring>

namespace c10 {

namespace detail {
  inline C10_HOST_DEVICE float f32_from_bits(uint16_t src) {
    float* res = 0;
    uint32_t tmp = src;
    tmp <<= 16;

    // We should be using memcpy in order to respect the strict aliasing rule
    // but it fails in the HIP environment.
    res = reinterpret_cast<float*>(&tmp);
    return *res;
  }

  inline C10_HOST_DEVICE uint16_t bits_from_f32(float src) {
    // We should be using memcpy in order to respect the strict aliasing rule
    // but it fails in the HIP environment.
    uint32_t* res = reinterpret_cast<uint32_t*>(&src);
    return *res >>= 16;
  }
} // namespace detail

struct alignas(2) BFloat16 {
  uint16_t val_;

  // HIP wants __host__ __device__ tag, CUDA does not
#ifdef __HIP_PLATFORM_HCC__
  C10_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;

#endif

  explicit inline C10_HOST_DEVICE BFloat16(float value);
  explicit inline C10_HOST_DEVICE operator float() const;
};

} // namespace c10


#include <c10/util/BFloat16-inl.h>
