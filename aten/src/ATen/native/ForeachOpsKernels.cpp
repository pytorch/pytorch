#include <ATen/ATen.h>
#include <ATen/native/ForeachUtils.h>

namespace at { namespace native {

std::vector<Tensor> foreach_tensor_exp_slow(TensorList tensors) {
  check_foreach_api_restrictions(tensors);

  std::vector<Tensor> result;
  for (const auto& t : tensors) {
    result.emplace_back(t.exp());
  }

  return result;
}

void foreach_tensor_exp_slow_(TensorList tensors) {
  check_foreach_api_restrictions(tensors);

  for (auto& t : tensors) {
    t.exp_();
  }
}

std::vector<Tensor> foreach_tensor_sqrt_slow(TensorList tensors) {
  check_foreach_api_restrictions(tensors);

  std::vector<Tensor> result;
  for (const auto& t : tensors) {
    result.emplace_back(t.sqrt());
  }

  return result;
}

void foreach_tensor_sqrt_slow_(TensorList tensors) {
  check_foreach_api_restrictions(tensors);

  for (auto& t : tensors) {
    t.sqrt_();
  }
}

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

std::vector<Tensor> foreach_tensor_sub_scalar_kernel_slow(TensorList tensors, Scalar scalar) {
  check_foreach_api_restrictions(tensors);

  std::vector<Tensor> result;
  for (const auto& t: tensors) {
    result.emplace_back(t.sub(scalar));
  }

  return result;
}

void foreach_tensor_sub_scalar_kernel_slow_(TensorList tensors, Scalar scalar) {
  check_foreach_api_restrictions(tensors);

  for (auto& t: tensors) {
    t.sub_(scalar);
  }
}

std::vector<Tensor> foreach_tensor_div_scalar_kernel_slow(TensorList tensors, Scalar scalar) {
  check_foreach_api_restrictions(tensors);

  std::vector<Tensor> result;
  for (const auto& t: tensors) {
    result.emplace_back(t.div(scalar));
  }

  return result;
}

void foreach_tensor_div_scalar_kernel_slow_(TensorList tensors, Scalar scalar) {
  check_foreach_api_restrictions(tensors);

  for (auto& t: tensors) {
    t.div_(scalar);
  }
}

std::vector<Tensor> foreach_tensor_mul_scalar_kernel_slow(TensorList tensors, Scalar scalar) {
  check_foreach_api_restrictions(tensors);

  std::vector<Tensor> result;
  for (const auto& t: tensors) {
    result.emplace_back(t.mul(scalar));
  }

  return result;
}

void foreach_tensor_mul_scalar_kernel_slow_(TensorList tensors, Scalar scalar) {
  check_foreach_api_restrictions(tensors);

  for (auto& t: tensors) {
    t.mul_(scalar);
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

std::vector<Tensor> foreach_tensor_sub_list_kernel_slow(TensorList tensors1, TensorList tensors2) {
  check_foreach_api_restrictions(tensors1, tensors2);

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    result.emplace_back(tensors1[i].sub(tensors2[i]));
  }

  return result;
}

void foreach_tensor_sub_list_kernel_slow_(TensorList tensors1, TensorList tensors2) {
  check_foreach_api_restrictions(tensors1, tensors2);

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].sub_(tensors2[i]);
  }
}

std::vector<Tensor> foreach_tensor_mul_list_kernel_slow(TensorList tensors1, TensorList tensors2) {
  check_foreach_api_restrictions(tensors1, tensors2);

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    result.emplace_back(tensors1[i].mul(tensors2[i]));
  }

  return result;
}

void foreach_tensor_mul_list_kernel_slow_(TensorList tensors1, TensorList tensors2) {
  check_foreach_api_restrictions(tensors1, tensors2);

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].mul_(tensors2[i]);
  }
}

std::vector<Tensor> foreach_tensor_div_list_kernel_slow(TensorList tensors1, TensorList tensors2) {
  check_foreach_api_restrictions(tensors1, tensors2);

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    result.emplace_back(tensors1[i].div(tensors2[i]));
  }

  return result;
}

void foreach_tensor_div_list_kernel_slow_(TensorList tensors1, TensorList tensors2) {
  check_foreach_api_restrictions(tensors1, tensors2);

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].div_(tensors2[i]);
  }
}

}} // namespace at::native
