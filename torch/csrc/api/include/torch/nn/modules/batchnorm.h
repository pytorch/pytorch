#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <cstdint>

namespace torch {
namespace nn {
struct BatchNormOptions {
  /* implicit */ BatchNormOptions(int64_t features);
  TORCH_ARG(int64_t, features);
  TORCH_ARG(bool, affine) = true;
  TORCH_ARG(bool, stateful) = false;
  TORCH_ARG(double, eps) = 1e-5;
  TORCH_ARG(double, momentum) = 0.1;
};

class BatchNormImpl : public torch::nn::Cloneable<BatchNormImpl> {
 public:
  explicit BatchNormImpl(BatchNormOptions options);

  void reset() override;

  Tensor forward(Tensor input);
  Tensor pure_forward(Tensor input, Tensor mean, Tensor variance);

  BatchNormOptions options;
  Tensor weight;
  Tensor bias;
  Tensor running_mean;
  Tensor running_variance;
};

TORCH_MODULE(BatchNorm);

} // namespace nn
} // namespace torch
