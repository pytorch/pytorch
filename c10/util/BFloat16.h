#pragma once

// Defines the bloat16 type (brain floating-point). This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.

#include <c10/macros/Macros.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>

namespace c10 {

namespace detail {
  inline float f32_from_bf16(uint16_t src) {
    float res = 0;
    uint16_t* tmp = reinterpret_cast<uint16_t*>(&res);
    tmp[0] = 0;
    tmp[1] = src;
    return res;
  }

  inline uint16_t bf16_from_f32(float src) {
    if (std::isnan(src)) {
      return 0x7FC0; // NaN
    }

    uint16_t* res = reinterpret_cast<uint16_t*>(&src);
    return res[1];
  }
} // namespace detail

struct alignas(2) BFloat16 {
  uint16_t val_;

  BFloat16() = default;

  inline C10_HOST_DEVICE BFloat16(float value);
  inline C10_HOST_DEVICE operator float() const;
};

} // namespace c10


#include <c10/util/BFloat16-inl.h>
