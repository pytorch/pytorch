// Functions that fill Tensors with constants. Implementations are in Fill.cpp.

#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at { namespace native {

DECLARE_DISPATCH(void(*)(TensorIterator&, const Scalar&), fill_stub);

}} // namespace at::native
