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

std::vector<Tensor> foreach_add_list_kernel_fallback(TensorList tensors1, TensorList tensors2) {
  verify_list(tensors1, tensors2);

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].add(tensors2[i]);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_add_list_kernel_fallback_(TensorList tensors1, TensorList tensors2) {
  verify_list(tensors1, tensors2);

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].add_(tensors2[i]);
  }

  return tensors1.vec();
}

}} // namespace at::native
