#include <c10/Half.h>
#include <fp16/fp16.h>

#include <iostream>

namespace c10 {

static_assert(
    std::is_standard_layout<Half>::value,
    "c10::Half must be standard layout.");

namespace detail {

// Host functions for converting between FP32 and FP16 formats

float halfbits2float(unsigned short h) {
  return fp16_ieee_to_fp32_value(h);
}

unsigned short float2halfbits(float src) {
  return fp16_ieee_from_fp32_value(src);
}

} // namespace detail

std::ostream& operator<<(std::ostream& out, const Half& value) {
  out << (float)value;
  return out;
}

} // namespace c10
