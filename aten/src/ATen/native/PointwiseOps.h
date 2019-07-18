// Ternary and higher-order pointwise operations
#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

struct TensorIterator;

namespace native {

using addcmul_fn = void (*)(TensorIterator&, Scalar scalar);

DECLARE_DISPATCH(addcmul_fn, addcmul_stub);
} // namespace native
} // namespace at
