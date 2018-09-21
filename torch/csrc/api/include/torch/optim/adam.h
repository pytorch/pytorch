#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/serialization.h>

#include <ATen/ATen.h>

#include <utility>
#include <vector>

namespace torch {
namespace optim {

struct AdamOptions {
  /* implicit */ AdamOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, beta1) = 0.9;
  TORCH_ARG(double, beta2) = 0.999;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(bool, amsgrad) = false;
};

class Adam : public Optimizer {
 public:
  template <typename ParameterContainer>
  explicit Adam(ParameterContainer&& parameters, const AdamOptions& options)
      : Optimizer(std::forward<ParameterContainer>(parameters)),
        options(options) {}

  void step() override;

  template <class Archive>
  void serialize(Archive& ar) {
#if defined(TORCH_USE_CEREAL)
    ar(CEREAL_NVP(step_buffers_),
       CEREAL_NVP(exp_average_buffers_),
       CEREAL_NVP(exp_average_sq_buffers_),
       CEREAL_NVP(max_exp_average_sq_buffers_));
#endif // defined(TORCH_USE_CEREAL)
  }

  AdamOptions options;

 private:
#if defined(TORCH_USE_CEREAL)
  friend class cereal::access;
#endif // defined(TORCH_USE_CEREAL)
  Adam() : options(0) {}

  std::vector<int64_t> step_buffers_;
  std::vector<Tensor> exp_average_buffers_;
  std::vector<Tensor> exp_average_sq_buffers_;
  std::vector<Tensor> max_exp_average_sq_buffers_;
};
} // namespace optim
} // namespace torch

#if defined(TORCH_USE_CEREAL)
CEREAL_REGISTER_TYPE(torch::optim::Adam);
CEREAL_REGISTER_POLYMORPHIC_RELATION(
    torch::optim::Optimizer,
    torch::optim::Adam);
#endif // defined(TORCH_USE_CEREAL)
