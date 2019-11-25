#pragma once

// Defines the bloat16 type (brain floating-point). This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.

#include <c10/macros/Macros.h>
#include <cmath>
#include <cstring>

namespace c10 {

namespace detail {
  inline C10_HOST_DEVICE float f32_from_bits(uint16_t src) {
    float res = 0;
    uint32_t tmp = src;
    tmp <<= 16;

#ifdef __HIP_PLATFORM_HCC__
    float* tempRes;

    // We should be using memcpy in order to respect the strict aliasing rule
    // but it fails in the HIP environment.
    tempRes = reinterpret_cast<float*>(&tmp);
    res = *tempRes;
#else
    std::memcpy(&res, &tmp, sizeof(tmp));
#endif

    return res;
  }

  inline C10_HOST_DEVICE uint16_t bits_from_f32(float src) {
    uint32_t res = 0;

#ifdef __HIP_PLATFORM_HCC__
    // We should be using memcpy in order to respect the strict aliasing rule
    // but it fails in the HIP environment.
    uint32_t* tempRes = reinterpret_cast<uint32_t*>(&src);
    res = *tempRes;
#else
    std::memcpy(&res, &src, sizeof(res));
#endif

    return res >> 16;
  }

  inline C10_HOST_DEVICE uint16_t round_to_nearest_even(float src) {
    if (std::isnan(src)) {
      return 0x7FC0;
    } else {
      union {
        uint32_t U32;
        float F32;
      };

      F32 = src;
      uint32_t rounding_bias = ((U32 >> 16) & 1) + 0x7FFF;
      return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
    }
  }
} // namespace detail

struct alignas(2) BFloat16 {
  uint16_t x;

  // HIP wants __host__ __device__ tag, CUDA does not
#ifdef __HIP_PLATFORM_HCC__
  C10_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;
#endif

  struct from_bits_t {};
  static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  constexpr C10_HOST_DEVICE BFloat16(unsigned short bits, from_bits_t) : x(bits){};
  inline C10_HOST_DEVICE BFloat16(float value);
  inline C10_HOST_DEVICE operator float() const;
};

} // namespace c10


#include <c10/util/BFloat16-inl.h>
