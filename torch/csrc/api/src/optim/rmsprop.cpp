#include <torch/optim/rmsprop.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {

RMSpropOptions::RMSpropOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

RMSprop::RMSprop(
    std::shared_ptr<nn::Module> model,
    const RMSpropOptions& options)
    : Optimizer(model), options_(options) {}

const RMSpropOptions& RMSprop::options() const noexcept {
  return options_;
}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/rmsprop.py
void RMSprop::step() {
  for (auto& parameter : model_->parameters()) {
    auto& name = parameter.key;
    auto& grad = parameter->grad();
    auto& p = parameter->data();
    if (!grad.defined()) {
      continue;
    }

    if (square_avg_buffer_.find(name) == square_avg_buffer_.end()) {
      square_avg_buffer_[name] = at::zeros_like(p);
      if (options_.momentum_) {
        momentum_buffer_[name] = at::zeros_like(p);
      }
      if (options_.centered_) {
        grad_avg_buffer_[name] = at::zeros_like(p);
      }
    }

    auto d_p = torch::autograd::as_variable_ref(grad).data();
    if (options_.weight_decay_ > 0) {
      d_p.add_(p, options_.weight_decay_);
    }

    auto& square_avg = square_avg_buffer_[name];
    square_avg.mul_(options_.alpha_).addcmul_(d_p, d_p, 1.0 - options_.alpha_);

    at::Tensor avg;
    if (options_.centered_) {
      auto& grad_avg = grad_avg_buffer_[name];
      grad_avg.mul_(options_.alpha_).add_(d_p, 1.0 - options_.alpha_);
      avg = square_avg.addcmul(grad_avg, grad_avg, -1.0)
                .sqrt()
                .add_(options_.eps_);
    } else {
      avg = square_avg.sqrt().add_(options_.eps_);
    }

    if (options_.momentum_ > 0) {
      auto& buf = momentum_buffer_[name];
      buf.mul_(options_.momentum_).addcdiv_(d_p, avg);
      p.add_(buf, -options_.learning_rate_);
    } else {
      p.addcdiv_(d_p, avg, -options_.learning_rate_);
    }
  }
}
} // namespace optim
} // namespace torch
