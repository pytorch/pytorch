#pragma once

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <functional>
#include <memory>

#define TORCH_AUTOGRAD_KWARG(CLS, TYP, NAME, DEFAULT, OPTION) \
  TYP NAME##_ = DEFAULT;                                      \
  CLS& NAME(TYP x = OPTION) {                                 \
    NAME##_ = x;                                              \
    return *this;                                             \
  }

namespace torch {
namespace optim {
class OptimizerImpl {
 public:
  OptimizerImpl(std::shared_ptr<nn::Module> model) : model_(model) {}
  virtual ~OptimizerImpl() = default;
  virtual void init_state() {}
  virtual at::Scalar step(std::function<at::Scalar()> closure = NoLoss) = 0;
  void zero_grad();

  void set_model(std::shared_ptr<nn::Module> model);
  at::Scalar static NoLoss();

 protected:
  OptimizerImpl() {}
  std::shared_ptr<nn::Module> model_;
};

template <class Derived>
class Optimizer : public OptimizerImpl {
 public:
  Optimizer(std::shared_ptr<nn::Module> model) : OptimizerImpl(model) {}

  template <typename ModuleType>
  Optimizer(nn::ModuleHolder<ModuleType> module) : Optimizer(module.get()) {}

  std::shared_ptr<Derived> make() const {
    auto ptr = std::make_shared<Derived>(*static_cast<const Derived*>(this));
    ptr->init_state();
    return ptr;
  }

 protected:
  Optimizer() {}
};

} // namespace optim
} // namespace torch
