#pragma once

#include <ATen/native/DispatchStub.h>

namespace c10 {
class Scalar;
}

namespace at {
struct TensorIterator;
struct TensorIteratorBase;
class TensorBase;
}

namespace at { namespace native {

using structured_activation_fn = void (*)(TensorIteratorBase&);
using structured_activation_backward_fn = void (*)(TensorIteratorBase&);

using activation_fn = void (*)(TensorIterator&);
using activation_backward_fn = void (*)(TensorIterator&);
using softplus_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&);
using softplus_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&);
using threshold_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&);
using hardtanh_backward_fn = void (*)(TensorIterator&, const c10::Scalar&, const c10::Scalar&);
using hardsigmoid_fn = void(*)(TensorIteratorBase&);
using hardsigmoid_backward_fn = void(*)(TensorIteratorBase&);
using hardswish_fn = void(*)(TensorIterator&);
using hardswish_backward_fn = void(*)(TensorIterator&);
using shrink_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using softshrink_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using shrink_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using elu_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&, const c10::Scalar&);
using elu_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&, const c10::Scalar&, const c10::Scalar&, bool);
using leaky_relu_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using leaky_relu_backward_fn = void (*)(TensorIteratorBase&, const c10::Scalar&);
using log_sigmoid_cpu_fn = void (*)(TensorBase&, TensorBase&, const TensorBase&);

DECLARE_DISPATCH(elu_fn, elu_stub);
DECLARE_DISPATCH(elu_backward_fn, elu_backward_stub);
DECLARE_DISPATCH(softplus_fn, softplus_stub);
DECLARE_DISPATCH(softplus_backward_fn, softplus_backward_stub);
DECLARE_DISPATCH(log_sigmoid_cpu_fn, log_sigmoid_cpu_stub);
DECLARE_DISPATCH(activation_backward_fn, log_sigmoid_backward_stub);
DECLARE_DISPATCH(threshold_fn, threshold_stub);
DECLARE_DISPATCH(structured_activation_fn, GeluKernel);
DECLARE_DISPATCH(structured_activation_backward_fn, GeluBackwardKernel);
DECLARE_DISPATCH(hardtanh_backward_fn, hardtanh_backward_stub);
DECLARE_DISPATCH(hardsigmoid_fn, hardsigmoid_stub);
DECLARE_DISPATCH(hardsigmoid_backward_fn, hardsigmoid_backward_stub);
DECLARE_DISPATCH(hardswish_fn, hardswish_stub);
DECLARE_DISPATCH(hardswish_backward_fn, hardswish_backward_stub);
DECLARE_DISPATCH(shrink_fn, hardshrink_stub);
DECLARE_DISPATCH(softshrink_fn, softshrink_stub);
DECLARE_DISPATCH(shrink_backward_fn, shrink_backward_stub);
DECLARE_DISPATCH(leaky_relu_fn, leaky_relu_stub);
DECLARE_DISPATCH(leaky_relu_backward_fn, leaky_relu_backward_stub);
DECLARE_DISPATCH(structured_activation_fn, glu_stub);
DECLARE_DISPATCH(activation_backward_fn, glu_backward_stub);
DECLARE_DISPATCH(structured_activation_fn, silu_stub);
DECLARE_DISPATCH(structured_activation_backward_fn, silu_backward_stub);
DECLARE_DISPATCH(structured_activation_fn, mish_stub);
DECLARE_DISPATCH(activation_backward_fn, mish_backward_stub);

} // namespace native

} // namespace at
