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
TORCH_AUTOGRAD_OPTIMIZER_CLASS(Adam) {
 public:
  Adam(std::shared_ptr<nn::Module> model, double lr)
      : Optimizer_CRTP(model), lr_(lr) {}

  template <typename ModuleType>
  Adam(nn::ModuleHolder<ModuleType> module_holder, double lr)
      : Adam(module_holder.get(), lr) {}

  TORCH_AUTOGRAD_KWARG(Adam, double, beta1, 0.9, 0.9);
  TORCH_AUTOGRAD_KWARG(Adam, double, beta2, 0.999, 0.999);
  TORCH_AUTOGRAD_KWARG(Adam, double, weight_decay, 0, 0);
  TORCH_AUTOGRAD_KWARG(Adam, double, eps, 1e-8, 1e-8);
  TORCH_AUTOGRAD_KWARG(Adam, bool, amsgrad, false, true);
  double lr_;
  at::Scalar step(std::function<at::Scalar()> closure = OptimizerImpl::NoLoss)
      override;
  void init_state() override;

  template <class Archive>
  void serialize(Archive & ar) {
    ar(CEREAL_NVP(step_buffer_),
       CEREAL_NVP(exp_avg_buffer_),
       CEREAL_NVP(exp_avg_sq_buffer_),
       CEREAL_NVP(max_exp_avg_sq_buffer_));
  }

 private:
  friend class cereal::access;
  Adam() {}
  std::unordered_map<std::string, int> step_buffer_;
  std::unordered_map<std::string, at::Tensor> exp_avg_buffer_;
  std::unordered_map<std::string, at::Tensor> exp_avg_sq_buffer_;
  std::unordered_map<std::string, at::Tensor> max_exp_avg_sq_buffer_;
};

} // namespace optim
} // namespace torch
