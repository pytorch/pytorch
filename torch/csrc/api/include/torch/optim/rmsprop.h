#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>

#include <ATen/ATen.h>

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>

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
    ar(CEREAL_NVP(square_average_buffers_));
    ar(CEREAL_NVP(momentum_buffers_));
    ar(CEREAL_NVP(grad_average_buffers_));
  }

 private:
  friend class cereal::access;
  RMSprop() : options(0) {}

  std::vector<Tensor> square_average_buffers_;
  std::vector<Tensor> momentum_buffers_;
  std::vector<Tensor> grad_average_buffers_;
};

} // namespace optim
} // namespace torch
