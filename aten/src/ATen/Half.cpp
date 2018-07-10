#include "ATen/Half.h"

#include "ATen/Tensor.h"
#include "ATen/Context.h"

#include <iostream>

namespace at {

static_assert(std::is_standard_layout<Half>::value, "at::Half must be standard layout.");

namespace detail {

// Host functions for converting between FP32 and FP16 formats

void halfbits2float_helper(unsigned short* src, float* res) {
    unsigned h = *src;
    unsigned sign = ((h >> 15) & 1);
    unsigned exponent = ((h >> 10) & 0x1f);
    unsigned mantissa = ((h & 0x3ff) << 13);

    if (exponent == 0x1f) {  /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) {  /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1;  /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }

    *(unsigned*)res = ((sign << 31) | (exponent << 23) | mantissa);
}

void float2halfbits_helper(float* src, unsigned short* dest) {
    unsigned x = *(unsigned*)src;
    unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
    unsigned sign, exponent, mantissa;

    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000) {
      *dest = 0x7fffU;
      return ;
    }
  
    sign = ((x >> 16) & 0x8000);
  
    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefff) {
      *dest = sign | 0x7c00U;
      return; 
    }
    if (u < 0x33000001) {
      *dest = (sign | 0x0000);
      return;
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

    *dest = (sign | (exponent << 10) | mantissa);  
}

float halfbits2float(unsigned short bits) {
  float value;
  halfbits2float_helper(&bits, &value);
  return value;
}

unsigned short float2halfbits(float value) {
  unsigned short bits;
  float2halfbits_helper(&value, &bits);
  return bits;
}

} // namespace detail

std::ostream& operator<<(std::ostream & out, const Half& value) {
  out << (float)value;
  return out;
}

} // namespace at
