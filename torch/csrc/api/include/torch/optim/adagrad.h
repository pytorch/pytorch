#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/tensor.h>

#include <ATen/ATen.h>

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>

#include <utility>
#include <vector>

namespace torch {
namespace optim {

struct AdagradOptions {
  AdagradOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, lr_decay) = 0;
  TORCH_ARG(double, weight_decay) = 0;
};

class Adagrad : public Optimizer {
 public:
  template <typename ParameterContainer>
  explicit Adagrad(
      ParameterContainer&& parameters,
      const AdagradOptions& options)
      : Optimizer(std::forward<ParameterContainer>(parameters)),
        options_(options),
        sum_(zero_buffers_like(parameters_)),
        step_(parameters_.size(), 0) {}

  void step() override;

  const AdagradOptions& options() const noexcept;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(CEREAL_NVP(sum_));
    ar(CEREAL_NVP(step_));
  }

 private:
  friend class cereal::access;
  Adagrad() : options_(0) {}

  AdagradOptions options_;

  std::vector<Tensor> sum_;
  std::vector<double> step_;
};
} // namespace optim
} // namespace torch
