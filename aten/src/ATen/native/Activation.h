#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>

namespace at {

struct TensorIterator;

namespace native {

using activation_fn = void (*)(TensorIterator&);
using activation_backward_fn = void (*)(TensorIterator&);
using softplus_fn = void (*)(TensorIterator&, Scalar, Scalar);
using softplus_backward_fn = void (*)(TensorIterator&, Scalar, Scalar);
using threshold_fn = void (*)(TensorIterator&, Scalar, Scalar);
using hardtanh_backward_fn = void (*)(TensorIterator&, Scalar, Scalar);
using hardsigmoid_fn = void(*)(TensorIterator&);
using hardsigmoid_backward_fn = void(*)(TensorIterator&);
using shrink_fn = void (*)(TensorIterator&, Scalar);
using shrink_backward_fn = void (*)(TensorIterator&, Scalar);
using elu_fn = void (*)(TensorIterator&, Scalar, Scalar, Scalar);
using leaky_relu_fn = void (*)(TensorIterator&, Scalar);
using leaky_relu_backward_fn = void (*)(TensorIterator&, Scalar);
using log_sigmoid_cpu_fn = void (*)(Tensor& , Tensor&, const Tensor& );

DECLARE_DISPATCH(elu_fn, elu_stub);
DECLARE_DISPATCH(elu_fn, elu_backward_stub);
DECLARE_DISPATCH(softplus_fn, softplus_stub);
DECLARE_DISPATCH(softplus_backward_fn, softplus_backward_stub);
DECLARE_DISPATCH(log_sigmoid_cpu_fn, log_sigmoid_cpu_stub);
DECLARE_DISPATCH(activation_backward_fn, log_sigmoid_backward_cpu_stub);
DECLARE_DISPATCH(threshold_fn, threshold_stub);
DECLARE_DISPATCH(activation_fn, GeluKernel);
DECLARE_DISPATCH(activation_backward_fn, GeluBackwardKernel);
DECLARE_DISPATCH(hardtanh_backward_fn, hardtanh_backward_stub);
DECLARE_DISPATCH(hardsigmoid_fn, hardsigmoid_stub);
DECLARE_DISPATCH(hardsigmoid_backward_fn, hardsigmoid_backward_stub);
DECLARE_DISPATCH(shrink_fn, hardshrink_stub);
DECLARE_DISPATCH(shrink_fn, softshrink_stub);
DECLARE_DISPATCH(shrink_backward_fn, shrink_backward_stub);
DECLARE_DISPATCH(leaky_relu_fn, leaky_relu_stub);
DECLARE_DISPATCH(leaky_relu_backward_fn, leaky_relu_backward_stub);
DECLARE_DISPATCH(activation_fn, glu_stub);
DECLARE_DISPATCH(activation_backward_fn, glu_backward_stub);

} // namespace native

} // namespace at
