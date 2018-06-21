#include <torch/optim/adagrad.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {

Adagrad::Adagrad(std::shared_ptr<nn::Module> model, double lr)
    : Optimizer(model), lr_(lr) {}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
at::Scalar Adagrad::step(std::function<at::Scalar()> closure) {
  at::Scalar loss = closure();
  for (auto& parameter : model_->parameters()) {
    auto& name = parameter.key;
    auto& grad = parameter->grad();
    auto& p = parameter->data();
    if (!grad.defined())
      continue;

    auto d_p = torch::autograd::as_variable_ref(grad).data();
    if (weight_decay_ > 0) {
      d_p.add_(p, weight_decay_);
    };
    auto& step = step_[name];
    step += 1.0;
    auto clr = lr_ / (1.0 + (step - 1.0) * lr_decay_);
    at::Tensor buf;
    if (sum_.find(name) == sum_.end()) {
      buf = sum_[name] = at::zeros_like(p);
    } else {
      buf = sum_[name];
    }

    buf.addcmul_(d_p, d_p, 1.0);
    at::Tensor std = buf.sqrt().add_(1e-10);
    p.addcdiv_(d_p, std, -clr);
  }
  return loss;
}
} // namespace optim
} // namespace torch
