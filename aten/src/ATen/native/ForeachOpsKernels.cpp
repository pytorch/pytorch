#include <ATen/ATen.h>
#include <ATen/native/ForeachUtils.h>

namespace at { namespace native {

std::vector<Tensor> foreach_add_scalar_kernel_fallback(TensorList tensors, Scalar scalar) {
  verify_list(tensors);

  std::vector<Tensor> result;
  for (const auto& t : tensors) {
    auto temp = t.add(scalar);
    result.emplace_back(temp);
  }
  return result;
}

std::vector<Tensor> foreach_add_scalar_kernel_fallback_(TensorList tensors, Scalar scalar) {
  verify_list(tensors);

  for (auto& t : tensors) {
    t.add_(scalar);
  }

  return tensors.vec();
}

}} // namespace at::native
