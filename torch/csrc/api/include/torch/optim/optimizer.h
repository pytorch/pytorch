#pragma once

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <functional>
#include <memory>

namespace torch {
namespace optim {
namespace detail {
class OptimizerBase {
 public:
  OptimizerBase(std::shared_ptr<nn::Module> model);
  virtual ~OptimizerBase() = default;

  virtual void zero_grad();

 protected:
  OptimizerBase() = default;

  std::shared_ptr<nn::Module> model_;
};
} // namespace detail

class Optimizer : public detail::OptimizerBase {
 public:
  using detail::OptimizerBase::OptimizerBase;
  virtual void step() = 0;
};

class LossClosureOptimizer : public detail::OptimizerBase {
 public:
  using LossClosure = std::function<at::Scalar()>;
  using detail::OptimizerBase::OptimizerBase;
  virtual at::Scalar step(LossClosure closure) = 0;
};

} // namespace optim
} // namespace torch
