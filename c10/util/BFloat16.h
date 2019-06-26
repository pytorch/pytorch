#pragma once

// Defines the bloat16 type (brain floating-point). This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.

#include <c10/macros/Macros.h>
#include <cmath>
#include <cstring>

namespace c10 {

namespace detail {
  inline bool isSmallEndian() {
    int num = 1;
    return *(char *)&num == 1;
  }

  inline float f32_from_bits(uint16_t src) {
    float res = 0;
    uint16_t* tmp = reinterpret_cast<uint16_t*>(&res);
    if (isSmallEndian()) {
      tmp[0] = 0;
      tmp[1] = src;
    } else {
      tmp[0] = src;
      tmp[1] = 0;
    }

    return res;
  }

  inline uint16_t bits_from_f32(float src) {
    uint16_t* res = reinterpret_cast<uint16_t*>(&src);
    return isSmallEndian() ? res[1] : res[0];
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

  struct from_bits_t {};
  static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  constexpr C10_HOST_DEVICE BFloat16(unsigned short bits, from_bits_t) : val_(bits){};
  explicit inline C10_HOST_DEVICE BFloat16(float value);
  explicit inline C10_HOST_DEVICE operator float() const;
};

} // namespace c10


#include <c10/util/BFloat16-inl.h>
