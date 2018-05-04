#pragma once

#include "torch/containers.h"
#include "torch/detail.h"

#include "cereal/access.hpp"
#include "cereal/cereal.hpp"

namespace torch {
class OptimizerImpl {
 public:
  OptimizerImpl(Container model) : model_(model) {}
  virtual ~OptimizerImpl() = default;
  virtual void init_state() {}
  virtual void step() = 0;
  void zero_grad();

  void set_model(Container model);

 protected:
  OptimizerImpl() {}
  Container model_;
};

template <class Derived>
class Optimizer_CRTP : public OptimizerImpl {
 public:
  Optimizer_CRTP(Container model) : OptimizerImpl(model) {}
  std::shared_ptr<Derived> make() const {
    auto ptr = std::make_shared<Derived>(*static_cast<const Derived*>(this));
    ptr->init_state();
    return ptr;
  }

 protected:
  Optimizer_CRTP() {}
};

AUTOGRAD_OPTIMIZER_CLASS(SGD) {
 public:
  SGD(Container model, double lr) : Optimizer_CRTP(model), lr_(lr) {}
  AUTOGRAD_KWARG(SGD, double, momentum, 0, 0);
  AUTOGRAD_KWARG(SGD, double, dampening, 0, 0);
  AUTOGRAD_KWARG(SGD, double, weight_decay, 0, 0);
  AUTOGRAD_KWARG(SGD, bool, nesterov, false, true);
  double lr_;
  void step() override;
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

AUTOGRAD_OPTIMIZER_CLASS(Adagrad) {
 public:
  Adagrad(Container model, double lr) : Optimizer_CRTP(model), lr_(lr) {}
  AUTOGRAD_KWARG(Adagrad, double, lr_decay, 0, 0);
  AUTOGRAD_KWARG(Adagrad, double, weight_decay, 0, 0);
  double lr_;
  void step() override;
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

AUTOGRAD_OPTIMIZER_CLASS(RMSprop) {
 public:
  RMSprop(Container model, double lr) : Optimizer_CRTP(model), lr_(lr) {}
  AUTOGRAD_KWARG(RMSprop, double, alpha, 0.99, 0.99);
  AUTOGRAD_KWARG(RMSprop, double, eps, 1e-8, 1e-8);
  AUTOGRAD_KWARG(RMSprop, double, weight_decay, 0, 0);
  AUTOGRAD_KWARG(RMSprop, double, momentum, 0, 0);
  AUTOGRAD_KWARG(RMSprop, bool, centered, false, true);

  double lr_;
  void step() override;
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

AUTOGRAD_OPTIMIZER_CLASS(Adam) {
 public:
  Adam(Container model, double lr) : Optimizer_CRTP(model), lr_(lr) {}
  AUTOGRAD_KWARG(Adam, double, beta1, 0.9, 0.9);
  AUTOGRAD_KWARG(Adam, double, beta2, 0.999, 0.999);
  AUTOGRAD_KWARG(Adam, double, weight_decay, 0, 0);
  AUTOGRAD_KWARG(Adam, double, eps, 1e-8, 1e-8);
  AUTOGRAD_KWARG(Adam, bool, amsgrad, false, true);
  double lr_;
  void step() override;
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
