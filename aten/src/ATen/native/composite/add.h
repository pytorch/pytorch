#pragma once
#include <ATen/ScalarOps.h>

namespace at {
namespace native {

template <typename OPS>
Tensor add(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return OPS::add(self, wrapped_scalar_tensor(other), alpha);
}

template <typename OPS>
Tensor& add_(Tensor& self, const Scalar& other, const Scalar& alpha) {
    return OPS::add_(self, wrapped_scalar_tensor(other), alpha);
}

} // namespace native
} // namespace at
