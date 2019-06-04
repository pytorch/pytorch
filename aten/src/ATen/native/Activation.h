#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>

namespace at {

struct TensorIterator;

namespace native {

using threshold_fn = void (*)(TensorIterator&, Scalar, Scalar);
using activation_fn = void (*)(TensorIterator* /* it */);
using activation_backward_fn =
    void (*)(const Tensor& /* dY */, const Tensor& /* X */, Tensor* /* dX */);

DECLARE_DISPATCH(threshold_fn, threshold_stub);
DECLARE_DISPATCH(activation_fn, GeluKernel);
DECLARE_DISPATCH(activation_backward_fn, GeluBackwardKernel);

} // namespace native

} // namespace at
