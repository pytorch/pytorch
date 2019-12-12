#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>

namespace at {

struct TensorIterator;

namespace native {

using activation_fn = void (*)(TensorIterator&);
using activation_backward_fn = void (*)(TensorIterator&);
using threshold_fn = void (*)(TensorIterator&, Scalar, Scalar);
using hardtanh_backward_fn = void (*)(TensorIterator&, Scalar, Scalar);
using shrink_fn = void (*)(TensorIterator&, Scalar);
using shrink_backward_fn = void (*)(TensorIterator&, Scalar);
using elu_fn = void (*)(TensorIterator&, Scalar, Scalar, Scalar);

DECLARE_DISPATCH(elu_fn, elu_stub);
DECLARE_DISPATCH(elu_fn, elu_backward_stub);
DECLARE_DISPATCH(threshold_fn, threshold_stub);
DECLARE_DISPATCH(activation_fn, GeluKernel);
DECLARE_DISPATCH(activation_backward_fn, GeluBackwardKernel);
DECLARE_DISPATCH(hardtanh_backward_fn, hardtanh_backward_stub);
DECLARE_DISPATCH(shrink_fn, hardshrink_stub);
DECLARE_DISPATCH(shrink_fn, softshrink_stub);
DECLARE_DISPATCH(shrink_backward_fn, shrink_backward_stub);

} // namespace native

} // namespace at
