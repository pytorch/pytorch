#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/Generator.h>
#include <stdexcept>

namespace at { namespace native {

using unary_fn = void(*)(Tensor&, const Tensor&);

DECLARE_DISPATCH(unary_fn, absImpl);
DECLARE_DISPATCH(unary_fn, acosImpl);
DECLARE_DISPATCH(unary_fn, asinImpl);
DECLARE_DISPATCH(unary_fn, atanImpl);
DECLARE_DISPATCH(unary_fn, ceilImpl);
DECLARE_DISPATCH(unary_fn, cosImpl);
// DECLARE_DISPATCH(unary_fn, coshImpl);
DECLARE_DISPATCH(unary_fn, erfImpl);
DECLARE_DISPATCH(unary_fn, erfcImpl);
DECLARE_DISPATCH(unary_fn, expImpl);
DECLARE_DISPATCH(unary_fn, expm1Impl);
DECLARE_DISPATCH(unary_fn, floorImpl);
DECLARE_DISPATCH(unary_fn, logImpl);
DECLARE_DISPATCH(unary_fn, log10Impl);
DECLARE_DISPATCH(unary_fn, log1pImpl);
DECLARE_DISPATCH(unary_fn, log2Impl);
DECLARE_DISPATCH(unary_fn, roundImpl);
DECLARE_DISPATCH(unary_fn, rsqrtImpl);
DECLARE_DISPATCH(unary_fn, sigmoidImpl);
DECLARE_DISPATCH(unary_fn, sinImpl);
// DECLARE_DISPATCH(unary_fn, sinhImpl);
DECLARE_DISPATCH(unary_fn, sqrtImpl);
DECLARE_DISPATCH(unary_fn, tanImpl);
DECLARE_DISPATCH(unary_fn, tanhImpl);
DECLARE_DISPATCH(unary_fn, truncImpl);

DECLARE_DISPATCH(void(*)(Tensor&, const double, Generator *), bernoulli_mkl_stub);


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
