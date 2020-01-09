#pragma once

#include <torch/nn/options/batchnorm.h>
#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

namespace detail {
inline Tensor batch_norm(const Tensor& input,
                         const Tensor& running_mean,
                         const Tensor& running_var,
                         Tensor weight,
                         Tensor bias,
                         bool training,
                         c10::optional<double> momentum,
                         double eps) {
  if (training) {
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
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum.value(),
    eps,
    at::globalContext().userEnabledCuDNN());
}
} // namespace detail

inline Tensor batch_norm(const Tensor& input, const Tensor& running_mean,
                         const Tensor& running_var, const BatchNormFuncOptions& options = {}) {
  return detail::batch_norm(
    input,
    running_mean,
    running_var,
    options.weight(),
    options.bias(),
    options.training(),
    options.momentum(),
    options.eps());
}

} // namespace functional
} // namespace nn
} // namespace torch
