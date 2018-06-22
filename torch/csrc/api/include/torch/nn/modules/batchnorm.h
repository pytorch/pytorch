#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <torch/csrc/autograd/variable.h>

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
  std::vector<Variable> forward(std::vector<Variable>);
  const BatchNormOptions& options() const noexcept;

 private:
  BatchNormOptions options_;
  Variable weight_;
  Variable bias_;
  Variable running_mean_;
  Variable running_variance_;
};

TORCH_MODULE(BatchNorm);

} // namespace nn
} // namespace torch
