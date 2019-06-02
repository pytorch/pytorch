#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>

namespace at {

struct TensorIterator;

namespace native {

using threshold_fn = void (*)(TensorIterator&, Scalar, Scalar);
using activation_fn = void (*)(const Tensor& /* X */, Tensor* /* Y */);

DECLARE_DISPATCH(threshold_fn, threshold_stub);
DECLARE_DISPATCH(activation_fn, GeluKernel);

} // namespace native

} // namespace at
