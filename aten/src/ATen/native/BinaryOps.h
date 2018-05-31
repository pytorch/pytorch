#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { struct TensorIterator; }

namespace at { namespace native {

using add_fn = void(*)(TensorIterator&, Scalar alpha);
using binary_fn = void(*)(TensorIterator&);

extern DispatchStub<add_fn> add_stub;
extern DispatchStub<add_fn> sub_stub;
extern DispatchStub<binary_fn> mul_stub;
extern DispatchStub<binary_fn> div_stub;

}} // namespace at::native
