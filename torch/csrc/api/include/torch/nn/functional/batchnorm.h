#pragma once

#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor batch_norm(const Tensor& input, const Tensor& running_mean,
                         const Tensor& running_var, const Tensor& weight = Tensor(),
                         const Tensor& bias = Tensor(), bool training = false,
                         double momentum = 0.1, double eps = 1e-5) {
  if (training) {
    auto size = input.sizes();
    int64_t size_prods = size[0];
    for (int i = 0; i < size.size() - 2; i++) {
      size_prods *= size[i + 2];
    }
    TORCH_CHECK(size_prods != 1,
                "Expected more than 1 value per channel when trainng");
  }

  return torch::batch_norm(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
    at::globalContext().userEnabledCuDNN());
}

} // namespace functional
} // namespace nn
} // namespace torch
