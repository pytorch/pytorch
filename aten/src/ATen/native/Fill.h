// Functions that fill Tensors with constants. Implementations are in Fill.cpp.

#pragma once

#include <ATen/native/DispatchStub.h>

namespace c10 {
class Scalar;
}

namespace at {
class Tensor;
struct TensorIterator;

namespace native {

DECLARE_DISPATCH(void(*)(TensorIterator&, const c10::Scalar&), fill_stub);

Tensor& fill_out(Tensor& self, const Scalar& value);

}} // namespace at::native
