#pragma once

#include <ATen/ATen.h>
#include <stdexcept>
#include "CapabilityDispatch.h"

namespace at { namespace native {

using unary_fn = void(*)(Tensor&, const Tensor&);

extern DispatchStub<unary_fn> absImpl;
extern DispatchStub<unary_fn> ceilImpl;
extern DispatchStub<unary_fn> floorImpl;
extern DispatchStub<unary_fn> roundImpl;
extern DispatchStub<unary_fn> sqrtImpl;
extern DispatchStub<unary_fn> truncImpl;

// Missing unary functions
// acos
// asin
// atan
// cos
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
// sin
// sinh
// tan
// tanh

}} // namespace at::native
