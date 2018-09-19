#include <torch/optim/rmsprop.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {

RMSpropOptions::RMSpropOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/rmsprop.py
void RMSprop::step() {
  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor p = parameters_.at(i);
    if (!p.grad().defined()) {
      continue;
    }

    if (options.weight_decay_ > 0) {
      p.grad() = p.grad() + options.weight_decay_ * p;
    }

    auto square_average = buffer_at(square_average_buffers_, i);
    square_average.mul_(options.alpha_)
        .addcmul_(p.grad(), p.grad(), 1.0 - options.alpha_);

    Tensor average;
    if (options.centered_ > 0) {
      auto& grad_average = buffer_at(grad_average_buffers_, i);
      grad_average.mul_(options.alpha_).add_(p.grad(), 1.0 - options.alpha_);
      average = square_average.addcmul(grad_average, grad_average, -1.0)
                    .sqrt()
                    .add_(options.eps_);
    } else {
      average = square_average.sqrt().add_(options.eps_);
    }

    NoGradGuard guard;
    if (options.momentum_ > 0) {
      auto& momentum = buffer_at(momentum_buffers_, i);
      momentum.mul_(options.momentum_).addcdiv_(p.grad(), average);
      p.add_(momentum, -options.learning_rate_);
    } else {
      p.addcdiv_(p.grad(), average, -options.learning_rate_);
    }
  }
}
} // namespace optim
} // namespace torch
