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

struct AdagradOptions {
  AdagradOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, lr_decay) = 0;
  TORCH_ARG(double, weight_decay) = 0;
};

class Adagrad : public Optimizer {
 public:
  Adagrad(std::shared_ptr<nn::Module> model, const AdagradOptions& options);

  template <typename ModuleType>
  Adagrad(
      nn::ModuleHolder<ModuleType> module_holder,
      const AdagradOptions& options)
      : Adagrad(module_holder.get(), options) {}

  at::Scalar step(std::function<at::Scalar()> closure = NoLoss) override;

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

  std::unordered_map<std::string, at::Tensor> sum_;
  std::unordered_map<std::string, double> step_;
};
} // namespace optim
} // namespace torch
