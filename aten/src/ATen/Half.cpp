#include "ATen/Half.h"

#include "ATen/Tensor.h"
#include "ATen/Context.h"

#include <TH/TH.h>
#include <iostream>

namespace at {

static_assert(std::is_standard_layout<Half>::value, "at::Half must be standard layout.");

namespace detail {

float halfbits2float(unsigned short bits) {
  float value;
  TH_halfbits2float(&bits, &value);
  return value;
}

unsigned short float2halfbits(float value) {
  unsigned short bits;
  TH_float2halfbits(&value, &bits);
  return bits;
}

} // namespace detail

std::ostream& operator<<(std::ostream & out, const Half& value) {
  out << (float)value;
  return out;
}

} // namespace at
