#pragma once

#include <torch/types.h>
#include <torch/cuda.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor batch_norm(const Tensor& input, const Tensor& running_mean,
                         const Tensor& running_var, const Tensor& weight,
                         const Tensor& bias, bool training,
                         double momentum, double eps) {
  if (training) {
    std::vector<int64_t> size = input.sizes().vec();
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
    torch::cuda::cudnn_is_available());
}

} // namespace functional
} // namespace nn
} // namespace torch
