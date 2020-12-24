// Ternary and higher-order pointwise operations
#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

struct TensorIterator;

namespace native {

using pointwise_fn = void (*)(TensorIterator&, Scalar scalar);
using pointwise_fn_beta = void (*)(TensorIterator&, Scalar scalar, double beta);

DECLARE_DISPATCH(pointwise_fn, addcmul_stub);
DECLARE_DISPATCH(pointwise_fn, addcdiv_stub);
DECLARE_DISPATCH(pointwise_fn_beta, smooth_l1_backward_stub);
DECLARE_DISPATCH(pointwise_fn, mse_backward_stub);

} // namespace native
} // namespace at
