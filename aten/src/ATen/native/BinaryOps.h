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
DECLARE_DISPATCH(binary_fn, lt_stub);
DECLARE_DISPATCH(binary_fn, le_stub);
DECLARE_DISPATCH(binary_fn, gt_stub);
DECLARE_DISPATCH(binary_fn, ge_stub);
DECLARE_DISPATCH(binary_fn, eq_stub);
DECLARE_DISPATCH(binary_fn, ne_stub);
DECLARE_DISPATCH(binary_fn, smooth_l1_stub);

}} // namespace at::native
