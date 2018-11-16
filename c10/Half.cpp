#include <c10/Half.h>

#include <iostream>

namespace c10 {

static_assert(
    std::is_standard_layout<Half>::value,
    "c10::Half must be standard layout.");

namespace detail {

// Host functions for converting between FP32 and FP16 formats

float halfbits2float(unsigned short h) {
  unsigned sign = ((h >> 15) & 1);
  unsigned exponent = ((h >> 10) & 0x1f);
  unsigned mantissa = ((h & 0x3ff) << 13);

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

  unsigned result_bit = (sign << 31) | (exponent << 23) | mantissa;

  // Reinterpret the result bit pattern as a float
  float result_float;
  std::memcpy(&result_float, &result_bit, sizeof(result_float));
  return result_float;
}

unsigned short float2halfbits(float src) {
  // Reinterpret the float as a bit pattern
  unsigned x;
  std::memcpy(&x, &src, sizeof(x));

  unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
  unsigned sign, exponent, mantissa;

  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000) {
    return 0x7fffU;
  }

  sign = ((x >> 16) & 0x8000);

  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefff) {
    return sign | 0x7c00U;
  }
  if (u < 0x33000001) {
    return (sign | 0x0000);
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

  return (sign | (exponent << 10) | mantissa);
}

} // namespace detail

std::ostream& operator<<(std::ostream& out, const Half& value) {
  out << (float)value;
  return out;
}

} // namespace c10
