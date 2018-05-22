#pragma once

#include <ATen/ATen.h>
#include <stdexcept>
#include "CapabilityDispatch.h"

namespace at {
namespace native {

using unary_fn = void (*)(Tensor&, const Tensor&);
using unary_fn_ = void (*)(Tensor&);

extern DispatchStub<unary_fn> absImpl;
extern DispatchStub<unary_fn> acosImpl;
extern DispatchStub<unary_fn> asinImpl;
extern DispatchStub<unary_fn> atanImpl;
extern DispatchStub<unary_fn> ceilImpl;
extern DispatchStub<unary_fn> cloneImpl;
extern DispatchStub<unary_fn> erfImpl;
extern DispatchStub<unary_fn> expImpl;
extern DispatchStub<unary_fn> expm1Impl;
extern DispatchStub<unary_fn> fracImpl;
extern DispatchStub<unary_fn> floorImpl;
extern DispatchStub<unary_fn> logImpl;
extern DispatchStub<unary_fn> log10Impl;
extern DispatchStub<unary_fn> log1pImpl;
extern DispatchStub<unary_fn> log2Impl;
extern DispatchStub<unary_fn> negImpl;
extern DispatchStub<unary_fn> reciprocalImpl;
extern DispatchStub<unary_fn> roundImpl;
extern DispatchStub<unary_fn> rsqrtImpl;
extern DispatchStub<unary_fn> sigmoidImpl;
extern DispatchStub<unary_fn> sqrtImpl;
extern DispatchStub<unary_fn> tanhImpl;
extern DispatchStub<unary_fn> truncImpl;

extern DispatchStub<unary_fn_> abs_Impl;
extern DispatchStub<unary_fn_> acos_Impl;
extern DispatchStub<unary_fn_> asin_Impl;
extern DispatchStub<unary_fn_> atan_Impl;
extern DispatchStub<unary_fn_> ceil_Impl;
extern DispatchStub<unary_fn_> clone_Impl;
extern DispatchStub<unary_fn_> erf_Impl;
extern DispatchStub<unary_fn_> exp_Impl;
extern DispatchStub<unary_fn_> expm1_Impl;
extern DispatchStub<unary_fn_> frac_Impl;
extern DispatchStub<unary_fn_> floor_Impl;
extern DispatchStub<unary_fn_> log_Impl;
extern DispatchStub<unary_fn_> log10_Impl;
extern DispatchStub<unary_fn_> log1p_Impl;
extern DispatchStub<unary_fn_> log2_Impl;
extern DispatchStub<unary_fn_> neg_Impl;
extern DispatchStub<unary_fn_> reciprocal_Impl;
extern DispatchStub<unary_fn_> round_Impl;
extern DispatchStub<unary_fn_> rsqrt_Impl;
extern DispatchStub<unary_fn_> sigmoid_Impl;
extern DispatchStub<unary_fn_> sqrt_Impl;
extern DispatchStub<unary_fn_> tanh_Impl;
extern DispatchStub<unary_fn_> trunc_Impl;

extern DispatchStub<unary_fn> cosImpl;
extern DispatchStub<unary_fn> coshImpl;
extern DispatchStub<unary_fn> sinImpl;
extern DispatchStub<unary_fn> sinhImpl;
extern DispatchStub<unary_fn> tanImpl;

extern DispatchStub<unary_fn_> cos_Impl;
extern DispatchStub<unary_fn_> cosh_Impl;
extern DispatchStub<unary_fn_> sin_Impl;
extern DispatchStub<unary_fn_> sinh_Impl;
extern DispatchStub<unary_fn_> tan_Impl;

extern DispatchStub<void (*)(Tensor&, Scalar&)> fillImpl;

extern DispatchStub<void (*)(Tensor&, const Tensor&, Scalar&, Scalar&)>
    clampImpl;
extern DispatchStub<void (*)(Tensor&, const Tensor&, Scalar&)> clampMinImpl;
extern DispatchStub<void (*)(Tensor&, const Tensor&, Scalar&)> clampMaxImpl;

extern DispatchStub<void (*)(Tensor&, Scalar&, Scalar&)> clamp_Impl;
extern DispatchStub<void (*)(Tensor&, Scalar&)> clampMin_Impl;
extern DispatchStub<void (*)(Tensor&, Scalar&)> clampMax_Impl;

// Missing unary functions
// digamma
// clone
// contiguous
// erfinv
// lgamma
// sign

} // namespace native
} // namespace at
