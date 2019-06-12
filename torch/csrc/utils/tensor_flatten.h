#pragma once

#include <ATen/core/functional.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/ATen.h>
#include <utility>

namespace torch { namespace utils {

inline at::Tensor flatten_dense_tensors(at::TensorList tensors) {
  static auto flatten = [](const at::Tensor &t) { return t.contiguous().view({-1}); };
  if (tensors.size() == 1)
    return flatten(tensors[0]);
  return at::cat(fmap(tensors, flatten));
}

inline std::vector<at::Tensor> unflatten_dense_tensors(const at::Tensor& flat, at::TensorList tensors) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(tensors.size());
  size_t offset = 0;
  for (const auto & tensor : tensors) {
    auto numel = tensor.numel();
    outputs.push_back(flat.narrow(0, offset, numel).view(tensor.sizes()));
    offset += numel;
  }
  return outputs;
}


struct TensorGroup {
  std::vector<at::Tensor> tensors;
  size_t size = 0;

  at::DeprecatedTypeProperties& type() {
    AT_ASSERT(!tensors.empty());
    return tensors[0].type();
  }
};

// Helper function that takes a list of tensors and splits them into tensor
// groups by the size limit and outputs these tensor groups. If the input
// tensors are of different tensor types, they will be split into different
// groups as well.
//
// Two options of splitting provided to the user,
//
// Imagine the size_limit is 256 and the list of input tensors are:
// tensor_a(fp16 - 128 bytes),
// tensor_b(fp32 - 256 bytes),
// tensor_c(fp16 - 128 bytes),
//
// when fine_grained == false:
// The function will read the list of tensors sequentially and accumulate
// enough tensors for each data type until the size_limit, therefore:
// it will output: {{tensor_a, tensor_c}, {tensor_b}}
//
// when fine_grained == true:
// The function will read the list of tensors sequentially and  accumulate
// enough tensors for all data types until the size_limit, and then split
// the accumulated tensors into different groups by data types, therefore:
// it will output: {{tensor_a}, {tensor_b}, {tensor_c}}
TORCH_API std::vector<TensorGroup> take_tensors(
    at::TensorList tensors,
    size_t size_limit,
    bool fine_grained = false);

TORCH_API void reorder_tensors_like(std::vector<at::Tensor>& tensors, at::TensorList order);

TORCH_API std::pair<at::Tensor, at::Tensor> flatten_sparse_tensors(at::TensorList tensors);

TORCH_API std::vector<at::Tensor> unflatten_sparse_tensors(
    const at::Tensor& flat_indices,
    const at::Tensor& flat_values,
    at::TensorList tensors);

}}
