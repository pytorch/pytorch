#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void add_kernel(TensorIterator& iter, const Scalar& alpha);

void sub_kernel(TensorIterator& iter, const Scalar& alpha);

void mul_kernel(TensorIterator& iter);

void div_kernel(TensorIterator& iter);

}}} // at::native::xpu
