#include <torch/optim/rmsprop.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {

RMSprop::RMSprop(std::shared_ptr<nn::Module> model, double lr)
    : Optimizer(model), lr_(lr) {}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/rmsprop.py
at::Scalar RMSprop::step(std::function<at::Scalar()> closure) {
  at::Scalar loss = closure();
  for (auto& parameter : model_->parameters()) {
    auto& name = parameter.key;
    auto& grad = parameter->grad();
    auto& p = parameter->data();
    if (!grad.defined())
      continue;

    if (square_avg_buffer_.find(name) == square_avg_buffer_.end()) {
      square_avg_buffer_[name] = at::zeros_like(p);
      if (momentum_) {
        momentum_buffer_[name] = at::zeros_like(p);
      };
      if (centered_) {
        grad_avg_buffer_[name] = at::zeros_like(p);
      };
    };

    auto d_p = torch::autograd::as_variable_ref(grad).data();
    if (weight_decay_ > 0) {
      d_p.add_(p, weight_decay_);
    };

    auto& square_avg = square_avg_buffer_[name];
    square_avg.mul_(alpha_).addcmul_(d_p, d_p, 1.0 - alpha_);

    at::Tensor avg;
    if (centered_) {
      auto& grad_avg = grad_avg_buffer_[name];
      grad_avg.mul_(alpha_).add_(d_p, 1.0 - alpha_);
      avg = square_avg.addcmul(grad_avg, grad_avg, -1.0).sqrt().add_(eps_);
    } else {
      avg = square_avg.sqrt().add_(eps_);
    };

    if (momentum_ > 0) {
      auto& buf = momentum_buffer_[name];
      buf.mul_(momentum_).addcdiv_(d_p, avg);
      p.add_(buf, -lr_);
    } else {
      p.addcdiv_(d_p, avg, -lr_);
    };
  }
  return loss;
}
} // namespace optim
} // namespace torch
