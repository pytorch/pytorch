#include <ATen/ATen.h>
namespace at { namespace native {

std::vector<Tensor> foreach_exp_fallback(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  std::vector<Tensor> result;
  for (auto& t : tensors) {
    auto temp = t.exp();
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_exp__fallback(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  for (auto& t : tensors) {
    t.exp_();
  }

  return tensors.vec();
}

std::vector<Tensor> foreach_sqrt_fallback(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  std::vector<Tensor> result;
  for (auto& t : tensors) {
    auto temp = t.sqrt();
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_sqrt__fallback(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  for (auto& t : tensors) {
    t.sqrt_();
  }

  return tensors.vec();
}

std::vector<Tensor> foreach_add_scalar_kernel_fallback(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  std::vector<Tensor> result;
  for (auto& t : tensors) {
    auto temp = t.add(scalar);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_add_scalar__kernel_fallback(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  for (auto& t : tensors) {
    t.add_(scalar);
  }

  return tensors.vec();
}

std::vector<Tensor> foreach_sub_scalar_kernel_fallback(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  std::vector<Tensor> result;
  for (auto& t: tensors) {
    auto temp = t.sub(scalar);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_sub_scalar__kernel_fallback(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  for (auto& t: tensors) {
    t.sub_(scalar);
  }

  return tensors.vec();
}

std::vector<Tensor> foreach_div_scalar_kernel_fallback(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  std::vector<Tensor> result;
  for (auto& t: tensors) {
    auto temp = t.div(scalar);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_div_scalar__kernel_fallback(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  for (auto& t: tensors) {
    t.div_(scalar);
  }

  return tensors.vec();
}

std::vector<Tensor> foreach_mul_scalar_kernel_fallback(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  std::vector<Tensor> result;
  for (auto& t: tensors) {
    auto temp = t.mul(scalar);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_mul_scalar__kernel_fallback(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");

  for (auto& t: tensors) {
    t.mul_(scalar);
  }

  return tensors.vec();
}

std::vector<Tensor> foreach_add_list_kernel_fallback(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].add(tensors2[i]);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_add_list__kernel_fallback(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].add_(tensors2[i]);
  }

  return tensors1.vec();
}

std::vector<Tensor> foreach_sub_list_kernel_fallback(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].sub(tensors2[i]);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_sub_list__kernel_fallback(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].sub_(tensors2[i]);
  }

  return tensors1.vec();
}

std::vector<Tensor> foreach_mul_list_kernel_fallback(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].mul(tensors2[i]);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_mul_list__kernel_fallback(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].mul_(tensors2[i]);
  }

  return tensors1.vec();
}

std::vector<Tensor> foreach_div_list_kernel_fallback(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].div(tensors2[i]);
    result.emplace_back(temp);
  }

  return result;
}

std::vector<Tensor> foreach_div_list__kernel_fallback(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() ==  tensors2.size(), "Tensor lists must be of the same length.");

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].div_(tensors2[i]);
  }

  return tensors1.vec();
}

}} // namespace at::native