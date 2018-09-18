#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/serialization.h>

#include <ATen/ATen.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace optim {

struct RMSpropOptions {
  RMSpropOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, alpha) = 0.99;
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, momentum) = 0;
  TORCH_ARG(bool, centered) = false;
};

class RMSprop : public Optimizer {
 public:
  template <typename ParameterContainer>
  explicit RMSprop(
      ParameterContainer&& parameters,
      const RMSpropOptions& options)
      : Optimizer(std::forward<ParameterContainer>(parameters)),
        options(options) {}

  void step() override;

  RMSpropOptions options;

  template <class Archive>
  void serialize(Archive& ar) {
#if defined(TORCH_USE_CEREAL)
    ar(CEREAL_NVP(square_average_buffers_));
    ar(CEREAL_NVP(momentum_buffers_));
    ar(CEREAL_NVP(grad_average_buffers_));
#endif // defined(TORCH_USE_CEREAL)
  }

 private:
#if defined(TORCH_USE_CEREAL)
  friend class cereal::access;
#endif // defined(TORCH_USE_CEREAL)
  RMSprop() : options(0) {}

  std::vector<Tensor> square_average_buffers_;
  std::vector<Tensor> momentum_buffers_;
  std::vector<Tensor> grad_average_buffers_;
};
} // namespace optim
} // namespace torch

#if defined(TORCH_USE_CEREAL)
CEREAL_REGISTER_TYPE(torch::optim::RMSprop);
CEREAL_REGISTER_POLYMORPHIC_RELATION(
    torch::optim::Optimizer,
    torch::optim::RMSprop);
#endif // defined(TORCH_USE_CEREAL)
