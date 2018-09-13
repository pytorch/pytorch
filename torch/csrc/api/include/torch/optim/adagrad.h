#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/serialization.h>
#include <torch/tensor.h>

#include <ATen/ATen.h>

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
        options(options) {}

  void step() override;

  AdagradOptions options;

  template <class Archive>
  void serialize(Archive& ar) {
#if defined(TORCH_USE_CEREAL)
    ar(CEREAL_NVP(sum_));
    ar(CEREAL_NVP(step_));
#endif // defined(TORCH_USE_CEREAL)
  }

 private:
#if defined(TORCH_USE_CEREAL)
  friend class cereal::access;
#endif // defined(TORCH_USE_CEREAL)
  Adagrad() : options(0) {}

  std::vector<Tensor> sum_;
  std::vector<int64_t> step_;
};
} // namespace optim
} // namespace torch

#if defined(TORCH_USE_CEREAL)
CEREAL_REGISTER_TYPE(torch::optim::Adagrad);
CEREAL_REGISTER_POLYMORPHIC_RELATION(
    torch::optim::Optimizer,
    torch::optim::Adagrad);
#endif // defined(TORCH_USE_CEREAL)
