#pragma once

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <functional>
#include <memory>

namespace torch {
namespace optim {
class Optimizer {
 public:
  Optimizer(std::shared_ptr<nn::Module> model) : model_(model) {}
  virtual ~Optimizer() = default;

  void zero_grad();
  virtual at::Scalar step(std::function<at::Scalar()> closure) = 0;

  at::Scalar static NoLoss();

 protected:
  Optimizer() = default;

  std::shared_ptr<nn::Module> model_;
};

} // namespace optim
} // namespace torch
