#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { struct TensorIterator; }

namespace at { namespace native {

using binary_fn_alpha = void(*)(TensorIterator&, Scalar alpha);
using binary_fn = void(*)(TensorIterator&);

DECLARE_DISPATCH(binary_fn_alpha, add_stub);
DECLARE_DISPATCH(binary_fn_alpha, sub_stub);
DECLARE_DISPATCH(binary_fn, mul_stub);
DECLARE_DISPATCH(binary_fn, div_stub);
DECLARE_DISPATCH(binary_fn, atan2_stub);
DECLARE_DISPATCH(binary_fn, logical_xor_stub);

}} // namespace at::native
