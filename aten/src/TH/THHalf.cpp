#include "THHalf.h"
#include "ATen/Half.h"

THHalf TH_float2half(float f) {
  THHalf h;
  h.x = ::at::detail::float2halfbits(f);
  return h;
}

TH_API float TH_half2float(THHalf h) {
    return ::at::detail::halfbits2float(h.x);
}