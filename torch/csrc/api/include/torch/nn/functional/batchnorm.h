#pragma once

#include <torch/nn/options/batchnorm.h>
#include <torch/types.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {
namespace functional {

inline Tensor batch_norm(const Tensor& input, const Tensor& running_mean,
                         const Tensor& running_var, const BatchNormFuncOptions& options = {}) {
  if (options.training()) {
    auto size = input.sizes();
    int64_t size_prods = size[0];
    for (size_t i = 0; i < size.size() - 2; i++) {
      size_prods *= size[i + 2];
    }
    TORCH_CHECK(size_prods != 1,
                "Expected more than 1 value per channel when training, got input size ", size);
  }

  return torch::batch_norm(
    input,
    options.weight(),
    options.bias(),
    running_mean,
    running_var,
    options.training(),
    options.momentum().value(),
    options.eps(),
    at::globalContext().userEnabledCuDNN());
}

} // namespace functional
} // namespace nn
} // namespace torch
