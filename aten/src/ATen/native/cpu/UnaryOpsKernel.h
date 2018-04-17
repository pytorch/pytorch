#pragma once
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <stdexcept>
#include "CapabilityDispatch.h"

namespace at { namespace native {

template <CPUCapability C>
struct ceilImplC {
  static void function(Tensor& result, const Tensor& self);
};
template <CPUCapability C>
struct floorImplC {
  static void function(Tensor& result, const Tensor& self);
};
template <CPUCapability C>
struct roundImplC {
  static void function(Tensor& result, const Tensor& self);
};
template <CPUCapability C>
struct truncImplC {
  static void function(Tensor& result, const Tensor& self);
};
template <CPUCapability C>
struct sqrtImplC {
  static void function(Tensor& result, const Tensor& self);
};

// Missing unary functions
// TODO: Add generic apply function for contiguous and non-contiguous tensors
// The goal here is to move more ops entirely into ATen and take advantage of
// automatic vectorization with file-specific flags
// acos
// asin
// atan
// cos
// cosh
// digamma
// erf
// erfinv
// exp
// expm1
// frac
// lgamma
// log1p
// log
// rsqrt
// sigmoid
// sin
// sinh
// tan
// tanh
// trunc

}} // namespace at::native
