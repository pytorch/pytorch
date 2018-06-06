#pragma once

#include "torch/csrc/utils/functional.h"
#include "torch/csrc/assertions.h"

#include <ATen/ATen.h>
#include <utility>

namespace torch { namespace utils {

inline at::Tensor flatten_dense_tensors(at::TensorList tensors) {
  if (tensors.size() == 1) {
    return tensors[0].reshape({-1});
  } else {
    int64_t total_numel = 0;
    for (const auto & tensor : tensors) {
      total_numel += tensor.numel();
    }
    auto flat = tensors[0].type().tensor({total_numel});
    int64_t offset = 0;
    for (const auto & tensor : tensors) {
      auto numel = tensor.numel();
      flat.narrow(0, offset, numel).view_as(tensor).copy_(tensor);
      offset += numel;
    }
    return flat;
  }
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

  at::Type& type() {
    TORCH_ASSERT(!tensors.empty());
    return tensors[0].type();
  }
};

std::vector<TensorGroup> take_tensors(at::TensorList tensors, size_t size_limit);
void reorder_tensors_like(std::vector<at::Tensor>& tensors, at::TensorList order);

std::pair<at::Tensor, at::Tensor> flatten_sparse_tensors(at::TensorList tensors);

std::vector<at::Tensor> unflatten_sparse_tensors(
    const at::Tensor& flat_indices,
    const at::Tensor& flat_values,
    at::TensorList tensors);

}}
