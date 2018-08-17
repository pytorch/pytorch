#pragma once

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/optim/optimizer.h>
#include <torch/tensor.h>

#include <ATen/ATen.h>

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>

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
    ar(CEREAL_NVP(momentum_buffers_));
  }

  SGDOptions options;

 private:
  friend class cereal::access;
  SGD() : options(0) {}

  std::vector<Tensor> momentum_buffers_;
  /// Counts how often `step()` is called, for dampening.
  size_t iteration_{0};
};
} // namespace optim
} // namespace torch
