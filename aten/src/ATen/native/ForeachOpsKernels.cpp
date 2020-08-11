#include <ATen/ATen.h>
namespace at { namespace native {

std::vector<Tensor> foreach_add_scalar_kernel_fallback(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  std::vector<Tensor> result;
  for (const auto& t : tensors) {
    auto temp = t.add(scalar);
    result.emplace_back(temp);
  }
  return result;
}

std::vector<Tensor> foreach_add_scalar_kernel_fallback_(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  for (auto& t : tensors) {
    t.add_(scalar);
  }

  return tensors.vec();
}

}} // namespace at::native
