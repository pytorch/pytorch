#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>

#include <ATen/ATen.h>

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace torch {
namespace optim {
class RMSprop : public Optimizer<RMSprop> {
 public:
  RMSprop(std::shared_ptr<nn::Module> model, double lr)
      : Optimizer(model), lr_(lr) {}

  template <typename ModuleType>
  RMSprop(nn::ModuleHolder<ModuleType> module_holder, double lr)
      : RMSprop(module_holder.get(), lr) {}

  TORCH_AUTOGRAD_KWARG(RMSprop, double, alpha, 0.99, 0.99);
  TORCH_AUTOGRAD_KWARG(RMSprop, double, eps, 1e-8, 1e-8);
  TORCH_AUTOGRAD_KWARG(RMSprop, double, weight_decay, 0, 0);
  TORCH_AUTOGRAD_KWARG(RMSprop, double, momentum, 0, 0);
  TORCH_AUTOGRAD_KWARG(RMSprop, bool, centered, false, true);

  double lr_;
  at::Scalar step(std::function<at::Scalar()> closure = OptimizerImpl::NoLoss)
      override;
  void init_state() override;

  template <class Archive>
  void serialize(Archive & ar) {
    ar(CEREAL_NVP(square_avg_buffer_));
    ar(CEREAL_NVP(momentum_buffer_));
    ar(CEREAL_NVP(grad_avg_buffer_));
  }

 private:
  friend class cereal::access;
  RMSprop() {}
  std::unordered_map<std::string, at::Tensor> square_avg_buffer_;
  std::unordered_map<std::string, at::Tensor> momentum_buffer_;
  std::unordered_map<std::string, at::Tensor> grad_avg_buffer_;
};

} // namespace optim
} // namespace torch
