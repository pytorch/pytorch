#pragma once

// Defines the bloat16 type (brain floating-point). This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.

#include <c10/macros/Macros.h>
#include <cmath>
#include <cstring>

namespace c10 {

namespace detail {
  inline float f32_from_bits(uint16_t src) {
    float res = 0;
    uint32_t tmp = src;
    tmp <<= 16;
    memcpy(&res, &tmp, sizeof(tmp));
    return res;
  }

  inline uint16_t bits_from_f32(float src) {
    uint32_t res;
    memcpy(&res, &src, sizeof(res));
    return res >>= 16;
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

  inline C10_HOST_DEVICE BFloat16(float value);
  inline C10_HOST_DEVICE operator float() const;
};

} // namespace c10


#include <c10/util/BFloat16-inl.h>
