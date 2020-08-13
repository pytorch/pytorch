#include <ATen/ATen.h>
#include <ATen/native/ForeachUtils.h>

namespace at { namespace native {

std::vector<Tensor> foreach_tensor_add_scalar_kernel_slow(TensorList tensors, Scalar scalar) {
  verify_list(tensors);

  std::vector<Tensor> result;
  for (const auto& t : tensors) {
    auto temp = t.add(scalar);
    result.emplace_back(temp);
  }

  return result;
}

void foreach_tensor_add_scalar_kernel_slow_(TensorList tensors, Scalar scalar) {
  verify_list(tensors);

  for (auto& t : tensors) {
    t.add_(scalar);
  }
}

std::vector<Tensor> foreach_tensor_sub_scalar_kernel_slow(TensorList tensors, Scalar scalar) {
  verify_list(tensors);

  std::vector<Tensor> result;
  for (const auto& t: tensors) {
    auto temp = t.sub(scalar);
    result.emplace_back(temp);
  }

  return result;
}

void foreach_tensor_sub_scalar_kernel_slow_(TensorList tensors, Scalar scalar) {
  verify_list(tensors);

  for (auto& t: tensors) {
    t.sub_(scalar);
  }
}

std::vector<Tensor> foreach_tensor_div_scalar_kernel_slow(TensorList tensors, Scalar scalar) {
  verify_list(tensors);

  std::vector<Tensor> result;
  for (const auto& t: tensors) {
    auto temp = t.div(scalar);
    result.emplace_back(temp);
  }

  return result;
}

void foreach_tensor_div_scalar_kernel_slow_(TensorList tensors, Scalar scalar) {
  verify_list(tensors);

  for (auto& t: tensors) {
    t.div_(scalar);
  }
}

std::vector<Tensor> foreach_tensor_mul_scalar_kernel_slow(TensorList tensors, Scalar scalar) {
  verify_list(tensors);

  std::vector<Tensor> result;
  for (const auto& t: tensors) {
    auto temp = t.mul(scalar);
    result.emplace_back(temp);
  }

  return result;
}

void foreach_tensor_mul_scalar_kernel_slow_(TensorList tensors, Scalar scalar) {
  verify_list(tensors);

  for (auto& t: tensors) {
    t.mul_(scalar);
  }
}

std::vector<Tensor> foreach_tensor_add_list_kernel_slow(TensorList tensors1, TensorList tensors2) {
  verify_list(tensors1, tensors2);

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].add(tensors2[i]);
    result.emplace_back(temp);
  }

  return result;
}

void foreach_tensor_add_list_kernel_slow_(TensorList tensors1, TensorList tensors2) {
  verify_list(tensors1, tensors2);

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].add_(tensors2[i]);
  }
}

std::vector<Tensor> foreach_tensor_sub_list_kernel_slow(TensorList tensors1, TensorList tensors2) {
  verify_list(tensors1, tensors2);

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].sub(tensors2[i]);
    result.emplace_back(temp);
  }

  return result;
}

void foreach_tensor_sub_list_kernel_slow_(TensorList tensors1, TensorList tensors2) {
  verify_list(tensors1, tensors2);

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].sub_(tensors2[i]);
  }
}

std::vector<Tensor> foreach_tensor_mul_list_kernel_slow(TensorList tensors1, TensorList tensors2) {
  verify_list(tensors1, tensors2);

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].mul(tensors2[i]);
    result.emplace_back(temp);
  }

  return result;
}

void foreach_tensor_mul_list_kernel_slow_(TensorList tensors1, TensorList tensors2) {
  verify_list(tensors1, tensors2);

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].mul_(tensors2[i]);
  }
}

std::vector<Tensor> foreach_tensor_div_list_kernel_slow(TensorList tensors1, TensorList tensors2) {
  verify_list(tensors1, tensors2);

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].div(tensors2[i]);
    result.emplace_back(temp);
  }

  return result;
}

void foreach_tensor_div_list_kernel_slow_(TensorList tensors1, TensorList tensors2) {
  verify_list(tensors1, tensors2);

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].div_(tensors2[i]);
  }
}

}} // namespace at::native
