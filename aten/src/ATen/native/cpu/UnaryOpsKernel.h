#pragma once
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <stdexcept>
#include "CapabilityDispatch.h"

namespace at { namespace native {

#define FUNCImplC(NAME) \
template <CPUCapability C>\
struct NAME ## ImplC {\
  static void function(Tensor& result, const Tensor& self);\
};\

#define UNARY_OPS_MACRO(MACRO) \
  MACRO (abs) \
  MACRO (ceil) \
  MACRO (cos) \
  MACRO (exp) \
  MACRO (floor) \
  MACRO (log) \
  MACRO (round) \
  MACRO (sin) \
  MACRO (sqrt) \
  MACRO (trunc) \

UNARY_OPS_MACRO(FUNCImplC)

// Missing unary functions
// TODO: Add generic apply function for contiguous and non-contiguous tensors
// The goal here is to move more ops entirely into ATen and take advantage of
// automatic vectorization with file-specific flags
// acos
// asin
// atan
// cosh
// digamma
// erf
// erfinv
// expm1
// frac
// lgamma
// log1p
// rsqrt
// sigmoid
// sinh
// tan
// tanh
// trunc

}} // namespace at::native
