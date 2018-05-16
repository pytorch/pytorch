#pragma once

#include <torch/nn/module.h>
#include "torch/detail.h"

#include "cereal/access.hpp"
#include "cereal/cereal.hpp"

namespace torch {
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
class Optimizer_CRTP : public OptimizerImpl {
 public:
  Optimizer_CRTP(std::shared_ptr<nn::Module> model) : OptimizerImpl(model) {}
  std::shared_ptr<Derived> make() const {
    auto ptr = std::make_shared<Derived>(*static_cast<const Derived*>(this));
    ptr->init_state();
    return ptr;
  }

 protected:
  Optimizer_CRTP() {}
};

TORCH_AUTOGRAD_OPTIMIZER_CLASS(LBFGS) {
 public:
  LBFGS(std::shared_ptr<nn::Module> model, double lr)
      : Optimizer_CRTP(model), lr_(lr) {}
  TORCH_AUTOGRAD_KWARG(LBFGS, int, max_iter, 20, 20);
  TORCH_AUTOGRAD_KWARG(LBFGS, int, max_eval, 25, 25);
  TORCH_AUTOGRAD_KWARG(LBFGS, float, tolerance_grad, 1e-5, 1e-5);
  TORCH_AUTOGRAD_KWARG(LBFGS, float, tolerance_change, 1e-9, 1e-9);
  TORCH_AUTOGRAD_KWARG(LBFGS, int, history_size, 100, 100);

  double lr_;
  at::Scalar step(std::function<at::Scalar()> closure) override;
  void init_state() override;

  template <class Archive>
  void serialize(Archive & ar) {
    ar(CEREAL_NVP(d));
    ar(CEREAL_NVP(t));
    ar(CEREAL_NVP(H_diag));
    ar(CEREAL_NVP(prev_flat_grad));
    ar(CEREAL_NVP(prev_loss));
    ar(CEREAL_NVP(old_dirs));
    ar(CEREAL_NVP(old_stps));
  }

 private:
  friend class cereal::access;
  LBFGS() {}
  Tensor gather_flat_grad();
  void add_grad(const at::Scalar& step_size, const at::Tensor& update);

  at::Tensor d, H_diag, prev_flat_grad;
  at::Scalar t, prev_loss;
  std::vector<at::Tensor> ro, al;
  std::deque<at::Tensor> old_dirs, old_stps;
  int func_evals, state_n_iter;
};

TORCH_AUTOGRAD_OPTIMIZER_CLASS(SGD) {
 public:
  SGD(std::shared_ptr<nn::Module> model, double lr)
      : Optimizer_CRTP(model), lr_(lr) {}
  TORCH_AUTOGRAD_KWARG(SGD, double, momentum, 0, 0);
  TORCH_AUTOGRAD_KWARG(SGD, double, dampening, 0, 0);
  TORCH_AUTOGRAD_KWARG(SGD, double, weight_decay, 0, 0);
  TORCH_AUTOGRAD_KWARG(SGD, bool, nesterov, false, true);
  double lr_;

  at::Scalar step(std::function<at::Scalar()> closure = OptimizerImpl::NoLoss) override;

  void init_state() override;

  template <class Archive>
  void serialize(Archive & ar) {
    ar(CEREAL_NVP(momentum_buffers_));
  }

 private:
  friend class cereal::access;
  SGD() {}
  std::unordered_map<std::string, at::Tensor> momentum_buffers_;
};

TORCH_AUTOGRAD_OPTIMIZER_CLASS(Adagrad) {
 public:
  Adagrad(std::shared_ptr<nn::Module> model, double lr)
      : Optimizer_CRTP(model), lr_(lr) {}
  TORCH_AUTOGRAD_KWARG(Adagrad, double, lr_decay, 0, 0);
  TORCH_AUTOGRAD_KWARG(Adagrad, double, weight_decay, 0, 0);
  double lr_;
  at::Scalar step(std::function<at::Scalar()> closure = OptimizerImpl::NoLoss) override;
  void init_state() override;

  template <class Archive>
  void serialize(Archive & ar) {
    ar(CEREAL_NVP(sum_));
    ar(CEREAL_NVP(step_));
  }

 private:
  friend class cereal::access;
  Adagrad() {}
  std::unordered_map<std::string, at::Tensor> sum_;
  std::unordered_map<std::string, double> step_;
};

TORCH_AUTOGRAD_OPTIMIZER_CLASS(RMSprop) {
 public:
  RMSprop(std::shared_ptr<nn::Module> model, double lr)
      : Optimizer_CRTP(model), lr_(lr) {}
  TORCH_AUTOGRAD_KWARG(RMSprop, double, alpha, 0.99, 0.99);
  TORCH_AUTOGRAD_KWARG(RMSprop, double, eps, 1e-8, 1e-8);
  TORCH_AUTOGRAD_KWARG(RMSprop, double, weight_decay, 0, 0);
  TORCH_AUTOGRAD_KWARG(RMSprop, double, momentum, 0, 0);
  TORCH_AUTOGRAD_KWARG(RMSprop, bool, centered, false, true);

  double lr_;
  at::Scalar step(std::function<at::Scalar()> closure = OptimizerImpl::NoLoss) override;
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

TORCH_AUTOGRAD_OPTIMIZER_CLASS(Adam) {
 public:
  Adam(std::shared_ptr<nn::Module> model, double lr)
      : Optimizer_CRTP(model), lr_(lr) {}
  TORCH_AUTOGRAD_KWARG(Adam, double, beta1, 0.9, 0.9);
  TORCH_AUTOGRAD_KWARG(Adam, double, beta2, 0.999, 0.999);
  TORCH_AUTOGRAD_KWARG(Adam, double, weight_decay, 0, 0);
  TORCH_AUTOGRAD_KWARG(Adam, double, eps, 1e-8, 1e-8);
  TORCH_AUTOGRAD_KWARG(Adam, bool, amsgrad, false, true);
  double lr_;
  at::Scalar step(std::function<at::Scalar()> closure = OptimizerImpl::NoLoss) override;
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

} // namespace torch
