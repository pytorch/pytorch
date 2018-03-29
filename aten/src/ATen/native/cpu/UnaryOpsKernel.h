#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <stdexcept>
#include "CapabilityDispatch.h"

namespace at { namespace native {

using unary_fn = void(*)(Tensor&, const Tensor&);

extern DispatchStub<unary_fn> absImpl;
extern DispatchStub<unary_fn> ceilImpl;
extern DispatchStub<unary_fn> cosImpl;
extern DispatchStub<unary_fn> expImpl;
extern DispatchStub<unary_fn> floorImpl;
extern DispatchStub<unary_fn> logImpl;
extern DispatchStub<unary_fn> roundImpl;
extern DispatchStub<unary_fn> sinImpl;
extern DispatchStub<unary_fn> sqrtImpl;
extern DispatchStub<unary_fn> truncImpl;

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

}} // namespace at::native
