#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

struct TensorIterator;

namespace native {

using pow_tensor_tensor_fn = void (*)(TensorIterator&);
using pow_tensor_scalar_fn = void (*)(TensorIterator&, Scalar);

DECLARE_DISPATCH(pow_tensor_tensor_fn, pow_tensor_tensor_stub);
DECLARE_DISPATCH(pow_tensor_scalar_fn, pow_tensor_scalar_stub);

} // namespace native

} // namespace at
