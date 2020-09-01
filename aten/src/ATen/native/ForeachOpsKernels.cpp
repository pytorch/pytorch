#include <ATen/ATen.h>
#include <ATen/native/ForeachUtils.h>

namespace at { namespace native {

std::vector<Tensor> foreach_tensor_add_scalar_kernel_slow(TensorList tensors, Scalar scalar) {
  check_foreach_api_restrictions(tensors);

  std::vector<Tensor> result;
  result.reserve(tensors.size());
  for (const auto& t : tensors) {
    result.emplace_back(t.add(scalar));
  }

  return result;
}

void foreach_tensor_add_scalar_kernel_slow_(TensorList tensors, Scalar scalar) {
  check_foreach_api_restrictions(tensors);

  for (auto& t : tensors) {
    t.add_(scalar);
  }
}

std::vector<Tensor> foreach_tensor_add_list_kernel_slow(TensorList tensors1, TensorList tensors2) {
  check_foreach_api_restrictions(tensors1, tensors2);

  std::vector<Tensor> result;
  result.reserve(tensors1.size());
  for (int i = 0; i < tensors1.size(); i++) {
    result.emplace_back(tensors1[i].add(tensors2[i]));
  }

  return result;
}

void foreach_tensor_add_list_kernel_slow_(TensorList tensors1, TensorList tensors2) {
  check_foreach_api_restrictions(tensors1, tensors2);

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].add_(tensors2[i]);
  }
}

}} // namespace at::native
