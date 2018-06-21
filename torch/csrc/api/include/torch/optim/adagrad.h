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
class Adagrad : public Optimizer {
 public:
  Adagrad(std::shared_ptr<nn::Module> model, double lr);

  template <typename ModuleType>
  Adagrad(nn::ModuleHolder<ModuleType> module_holder, double lr)
      : Adagrad(module_holder.get(), lr) {}

  TORCH_AUTOGRAD_KWARG(Adagrad, double, lr_decay, 0, 0);
  TORCH_AUTOGRAD_KWARG(Adagrad, double, weight_decay, 0, 0);
  double lr_;
  at::Scalar step(std::function<at::Scalar()> closure = NoLoss) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(CEREAL_NVP(sum_));
    ar(CEREAL_NVP(step_));
  }

 private:
  friend class cereal::access;
  Adagrad() {}
  std::unordered_map<std::string, at::Tensor> sum_;
  std::unordered_map<std::string, double> step_;
};
} // namespace optim
} // namespace torch
