#pragma once

/// Defines the bloat16 type (brain floating-point). This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.

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

struct alignas(2) bfloat16 {
  using underlying = uint16_t;
  uint16_t val_;
  explicit bfloat16() {}
  explicit bfloat16(uint16_t val) : val_(val) {}
  explicit bfloat16(float val) {
    val_ = detail::bf16_from_f32(val);
  }
};

} // namespace c10
