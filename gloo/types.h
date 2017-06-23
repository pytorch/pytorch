#pragma once

#include <iostream>

#ifdef __CUDA_ARCH__
#include <cuda_fp16.h>
#endif

#include "gloo/common/common.h"

namespace gloo {

struct float16;
float16 cpu_float2half_rn(float f);
float cpu_half2float(float16 h);

struct __attribute__((__aligned__(2))) float16 {
  uint16_t x;

  float16() : x(0) {}

  explicit float16(int val) {
    float16 res = cpu_float2half_rn(static_cast<float>(val));
    x = res.x;
  }
  explicit float16(unsigned long val) {
    float16 res = cpu_float2half_rn(static_cast<float>(val));
    x = res.x;
  }
  float16& operator=(const int& rhs) {
    float16 res = cpu_float2half_rn(static_cast<float>(rhs));
    x = res.x;
    return *this;
  }
  float16& operator=(const float16& rhs) {
    if (rhs != *this) {
      x = rhs.x;
    }
    return *this;
  }

  bool operator==(const float16& rhs) const {
    return x == rhs.x;
  }

  bool operator!=(const float16& rhs) const {
    return !(*this == rhs.x);
  }

  bool operator==(const int& rhs) const {
    float16 res = cpu_float2half_rn(static_cast<float>(rhs));
    return x == res.x;
  }
  bool operator==(const unsigned long& rhs) const {
    float16 res = cpu_float2half_rn(static_cast<float>(rhs));
    return x == res.x;
  }
#ifdef __CUDA_ARCH__
  float16(half h) {
#if CUDA_VERSION >= 9000
    x = reinterpret_cast<__half_raw*>(&h)->x;
#else
    x = h.x;
#endif // CUDA_VERSION
  }

  // half and float16 are supposed to have identical representation so implicit
  // conversion should be fine
  /* implicit */
  operator half() const {
#if CUDA_VERSION >= 9000
    __half_raw hr;
    hr.x = this->x;
    return half(hr);
#else
    return (half) * this;
#endif // CUDA_VERSION
  }
#endif // __CUDA_ARCH

  float16& operator+=(const float16& rhs) {
    float r = cpu_half2float(*this) + cpu_half2float(rhs);
    *this = cpu_float2half_rn(r);
    return *this;
  }

  float16& operator*=(const float16& rhs) {
    float r = cpu_half2float(*this) * cpu_half2float(rhs);
    *this = cpu_float2half_rn(r);
    return *this;
  }

};

inline std::ostream& operator<<(std::ostream& stream, const float16& val) {
  stream << cpu_half2float(val);
  return stream;
}

inline float16 operator+(const float16& lhs, const float16& rhs) {
  float16 result = lhs;
  result += rhs;
  return result;
}

inline float16 operator*(const float16& lhs, const float16& rhs) {
  float16 result = lhs;
  result *= rhs;
  return result;
}

inline bool operator<(const float16& lhs, const float16& rhs) {
  return cpu_half2float(lhs) < cpu_half2float(rhs);
}

inline bool operator>(const float16& lhs, const float16& rhs) {
  return cpu_half2float(lhs) > cpu_half2float(rhs);
}

inline float16 cpu_float2half_rn(float f) {
  float16 ret;

  static_assert(
      sizeof(unsigned int) == sizeof(float),
      "Programming error sizeof(unsigned int) != sizeof(float)");

  unsigned* xp = reinterpret_cast<unsigned int*>(&f);
  unsigned x = *xp;
  unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
  unsigned sign, exponent, mantissa;

  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000) {
    ret.x = 0x7fffU;
    return ret;
  }

  sign = ((x >> 16) & 0x8000);

  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefff) {
    ret.x = sign | 0x7c00U;
    return ret;
  }
  if (u < 0x33000001) {
    ret.x = (sign | 0x0000);
    return ret;
  }

  exponent = ((u >> 23) & 0xff);
  mantissa = (u & 0x7fffff);

  if (exponent > 0x70) {
    shift = 13;
    exponent -= 0x70;
  } else {
    shift = 0x7e - exponent;
    exponent = 0;
    mantissa |= 0x800000;
  }
  lsb = (1 << shift);
  lsb_s1 = (lsb >> 1);
  lsb_m1 = (lsb - 1);

  // Round to nearest even.
  remainder = (mantissa & lsb_m1);
  mantissa >>= shift;
  if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
    ++mantissa;
    if (!(mantissa & 0x3ff)) {
      ++exponent;
      mantissa = 0;
    }
  }

  ret.x = (sign | (exponent << 10) | mantissa);

  return ret;
}

inline float cpu_half2float(float16 h) {
  unsigned sign = ((h.x >> 15) & 1);
  unsigned exponent = ((h.x >> 10) & 0x1f);
  unsigned mantissa = ((h.x & 0x3ff) << 13);

  if (exponent == 0x1f) { /* NaN or Inf */
    mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
    exponent = 0xff;
  } else if (!exponent) { /* Denorm or Zero */
    if (mantissa) {
      unsigned int msb;
      exponent = 0x71;
      do {
        msb = (mantissa & 0x400000);
        mantissa <<= 1; /* normalize */
        --exponent;
      } while (!msb);
      mantissa &= 0x7fffff; /* 1.mantissa is implicit */
    }
  } else {
    exponent += 0x70;
  }

  unsigned temp = ((sign << 31) | (exponent << 23) | mantissa);

  void* rp = &temp;
  return *(float*) rp;
}
}
