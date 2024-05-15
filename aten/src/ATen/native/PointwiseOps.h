// Ternary and higher-order pointwise operations
#pragma once

#include <ATen/native/DispatchStub.h>

namespace c10 {
class Scalar;
}

namespace at {

struct TensorIterator;
struct TensorIteratorBase;

namespace native {

using pointwise_fn = void (*)(TensorIterator&, const Scalar& scalar);
using structured_pointwise_fn = void (*)(TensorIteratorBase&, const Scalar& scalar);
using pointwise_fn_double = void (*)(TensorIterator&, const Scalar&, double);

DECLARE_DISPATCH(structured_pointwise_fn, addcmul_stub);
DECLARE_DISPATCH(structured_pointwise_fn, addcdiv_stub);
DECLARE_DISPATCH(pointwise_fn_double, smooth_l1_backward_stub);
DECLARE_DISPATCH(pointwise_fn_double, huber_backward_stub);
DECLARE_DISPATCH(pointwise_fn, mse_backward_stub);

} // namespace native
} // namespace at
