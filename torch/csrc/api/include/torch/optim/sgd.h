#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/serialization.h>
#include <torch/tensor.h>

#include <ATen/ATen.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch {
namespace optim {

struct SGDOptions {
  /* implicit */ SGDOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, momentum) = 0;
  TORCH_ARG(double, dampening) = 0;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(bool, nesterov) = false;
};

class SGD : public Optimizer {
 public:
  template <typename ParameterContainer>
  explicit SGD(ParameterContainer&& parameters, const SGDOptions& options)
      : Optimizer(std::forward<ParameterContainer>(parameters)),
        options(options) {}

  void step() override;

  template <class Archive>
  void serialize(Archive& ar) {
#if defined(TORCH_USE_CEREAL)
    ar(CEREAL_NVP(momentum_buffers_));
#endif // defined(TORCH_USE_CEREAL)
  }

  SGDOptions options;

 private:
#if defined(TORCH_USE_CEREAL)
  friend class cereal::access;
#endif // defined(TORCH_USE_CEREAL)
  SGD() : options(0) {}

  std::vector<Tensor> momentum_buffers_;
  /// Counts how often `step()` is called, for dampening.
  size_t iteration_{0};
};
} // namespace optim
} // namespace torch

#if defined(TORCH_USE_CEREAL)
CEREAL_REGISTER_TYPE(torch::optim::SGD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(
    torch::optim::Optimizer,
    torch::optim::SGD);
#endif // defined(TORCH_USE_CEREAL)
