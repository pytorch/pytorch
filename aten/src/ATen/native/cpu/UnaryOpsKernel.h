#pragma once

#include <ATen/ATen.h>
#include <stdexcept>
#include "CapabilityDispatch.h"

namespace at { namespace native {

using unary_fn = void(*)(Tensor&, const Tensor&);

extern DispatchStub<unary_fn> absImpl;
extern DispatchStub<unary_fn> acosImpl;
extern DispatchStub<unary_fn> asinImpl;
extern DispatchStub<unary_fn> atanImpl;
extern DispatchStub<unary_fn> ceilImpl;
extern DispatchStub<unary_fn> erfImpl;
extern DispatchStub<unary_fn> expImpl;
extern DispatchStub<unary_fn> expm1Impl;
extern DispatchStub<unary_fn> fracImpl;
extern DispatchStub<unary_fn> floorImpl;
extern DispatchStub<unary_fn> logImpl;
extern DispatchStub<unary_fn> log10Impl;
extern DispatchStub<unary_fn> log1pImpl;
extern DispatchStub<unary_fn> log2Impl;
extern DispatchStub<unary_fn> roundImpl;
extern DispatchStub<unary_fn> rsqrtImpl;
extern DispatchStub<unary_fn> sqrtImpl;
extern DispatchStub<unary_fn> truncImpl;


// Missing unary functions
// digamma
// lgamma

// TODO: See below
// erfinv
// fill
// frac
// clone
// contiguous
// clamp/_min/_max
// neg
// reciprocal
// sigmoid
// sign
// zero


}} // namespace at::native
