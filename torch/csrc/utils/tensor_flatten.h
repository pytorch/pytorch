#pragma once

#include "torch/csrc/utils/functional.h"
#include "torch/csrc/assertions.h"

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
  std::size_t offset = 0;
  for (const auto & tensor : tensors) {
    auto numel = tensor.numel();
    outputs.push_back(flat.narrow(0, offset, numel).view(tensor.sizes()));
    offset += numel;
  }
  return outputs;
}


struct TensorGroup {
  std::vector<at::Tensor> tensors;
  std::size_t size = 0;

  at::Type& type() {
    TORCH_ASSERT(!tensors.empty());
    return tensors[0].type();
  }
};

std::vector<TensorGroup> take_tensors(at::TensorList tensors, std::size_t size_limit);
void reorder_tensors_like(std::vector<at::Tensor>& tensors, at::TensorList order);

std::pair<at::Tensor, at::Tensor> flatten_sparse_tensors(at::TensorList tensors);

std::vector<at::Tensor> unflatten_sparse_tensors(
    const at::Tensor& flat_indices,
    const at::Tensor& flat_values,
    at::TensorList tensors);

}}
