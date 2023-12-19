#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add.h>
#include <ATen/ops/lerp_native.h>
#endif

namespace at::native {
TORCH_IMPL_FUNC(lerp_Tensor_mps)(const Tensor& self, const Tensor& end, const Tensor& weight, const Tensor& out) {
  // TODO: Write a much better implementation
  at::add_out(const_cast<Tensor&>(out), self, weight.mul(end.sub(self)));
}

} // namespace at::native
