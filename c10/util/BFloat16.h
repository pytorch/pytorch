#pragma once

// Defines the bloat16 type (brain floating-point). This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.

#include <c10/macros/Macros.h>
#include <cmath>
#include <cstring>
#include <iostream>

namespace c10 {

struct alignas(2) BFloat16 {
  uint16_t val_;

  struct from_bits_t {};
  static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  // HIP wants __host__ __device__ tag, CUDA does not
#ifdef __HIP_PLATFORM_HCC__
  C10_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;
#endif

  constexpr C10_HOST_DEVICE BFloat16(uint16_t bits, from_bits_t) : val_(bits){};

  inline C10_HOST_DEVICE BFloat16(float value) {
    uint32_t res;
  // TEST
  #ifdef __HIP_PLATFORM_HCC__
    std::memcpy(&res, &value, sizeof(res));
  #else
    std::memcpy(&res, &value, sizeof(res));
  #endif
    val_ = res >> 16;
  }

  inline C10_HOST_DEVICE float toFloat() const {
    float res = 0;
    uint32_t tmp = val_;
    tmp <<= 16;
    std::memcpy(&res, &tmp, sizeof(tmp));
    return res;
  }

  inline C10_HOST_DEVICE operator float() const {
    return this->toFloat();
  }
};

} // namespace c10
