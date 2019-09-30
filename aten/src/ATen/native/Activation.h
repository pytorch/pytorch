#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>

namespace at {

struct TensorIterator;

namespace native {

using threshold_fn = void (*)(TensorIterator&, Scalar, Scalar);
using activation_fn = void (*)(const Tensor& /* X */, Tensor* /* Y */);
using activation_backward_fn =
    void (*)(const Tensor& /* dY */, const Tensor& /* X */, Tensor* /* dX */);
using hardshrink_cpu_fn = void (*)(TensorIterator&, Scalar);
using hardshrink_backward_cpu_fn = void (*)(TensorIterator&, Scalar);

DECLARE_DISPATCH(threshold_fn, threshold_stub);
DECLARE_DISPATCH(activation_fn, GeluKernel);
DECLARE_DISPATCH(activation_backward_fn, GeluBackwardKernel);
DECLARE_DISPATCH(hardshrink_cpu_fn, hardshrink_cpu_stub);
DECLARE_DISPATCH(hardshrink_backward_cpu_fn, hardshrink_backward_cpu_stub);

} // namespace native

} // namespace at
