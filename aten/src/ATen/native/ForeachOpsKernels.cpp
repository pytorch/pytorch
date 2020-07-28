#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

std::vector<Tensor> foreach_add_scalar_kernel_cpu(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(std::all_of(tensors.begin(), tensors.end(), [] (const Tensor& t) {
    return t.layout() == at::kStrided;
  }), "Only tensors with strided layouts are supported.");

  TORCH_CHECK(std::all_of(tensors.begin(), tensors.end(), [] (const Tensor& t) {
    return t.is_non_overlapping_and_dense();
  }), "Only non overlapping and dense tensors are supported.");

  std::vector<Tensor> result;
  for (int i = 0; i < tensors.size(); i++) {
    auto temp = tensors[i].add(scalar);
    result.emplace_back(temp);
  }
  return result;
}

std::vector<Tensor> foreach_add_scalar__kernel_cpu(TensorList tensors, Scalar scalar) {
  TORCH_CHECK(std::all_of(tensors.begin(), tensors.end(), [] (const Tensor& t) {
    return t.layout() == at::kStrided;
  }), "Only tensors with strided layouts are supported.");

  TORCH_CHECK(std::all_of(tensors.begin(), tensors.end(), [] (const Tensor& t) {
    return t.is_non_overlapping_and_dense();
  }), "Only non overlapping and dense tensors are supported.");

  for (int i = 0; i < tensors.size(); i++) {
    tensors[i].add_(scalar);
  }

  return tensors.vec();
}

std::vector<Tensor> foreach_add_list_kernel_cpu(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(std::all_of(tensors1.begin(), tensors1.end(), [] (const Tensor& t) {
    return t.layout() == at::kStrided;
  }), "Only tensors with strided layouts are supported.");

  TORCH_CHECK(std::all_of(tensors1.begin(), tensors1.end(), [] (const Tensor& t) {
    return t.is_non_overlapping_and_dense();
  }), "Only non overlapping and dense tensors are supported.");

  //[TODO]: checks.

  std::vector<Tensor> result;
  for (int i = 0; i < tensors1.size(); i++) {
    auto temp = tensors1[i].add(tensors2[i]);
    result.emplace_back(temp);
  }
  return result;
}

std::vector<Tensor> foreach_add_list__kernel_cpu(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(std::all_of(tensors1.begin(), tensors1.end(), [] (const Tensor& t) {
    return t.layout() == at::kStrided;
  }), "Only tensors with strided layouts are supported.");

  TORCH_CHECK(std::all_of(tensors1.begin(), tensors1.end(), [] (const Tensor& t) {
    return t.is_non_overlapping_and_dense();
  }), "Only non overlapping and dense tensors are supported.");

  //[TODO]: checks.

  for (int i = 0; i < tensors1.size(); i++) {
    tensors1[i].add_(tensors2[i]);
  }
  return tensors1.vec();
}

}} // namespace at::native
