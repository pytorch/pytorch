#include <torch/optim/rmsprop.h>

#include <torch/csrc/autograd/variable.h>

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
    auto& grad = parameters_.at(i).grad();
    auto& p = parameters_.at(i).data();
    if (!grad.defined()) {
      continue;
    }

    auto d_p = torch::autograd::as_variable_ref(grad).data();
    if (options.weight_decay_ > 0) {
      d_p.add_(p, options.weight_decay_);
    }

    auto square_average = buffer_at(square_average_buffers_, i).data();
    square_average.mul_(options.alpha_)
        .addcmul_(d_p, d_p, 1.0 - options.alpha_);

    at::Tensor average;
    if (options.centered_ > 0) {
      auto grad_average = buffer_at(grad_average_buffers_, i).data();
      grad_average.mul_(options.alpha_).add_(d_p, 1.0 - options.alpha_);
      average = square_average.addcmul(grad_average, grad_average, -1.0)
                    .sqrt()
                    .add_(options.eps_);
    } else {
      average = square_average.sqrt().add_(options.eps_);
    }

    if (options.momentum_ > 0) {
      auto momentum = buffer_at(momentum_buffers_, i).data();
      momentum.mul_(options.momentum_).addcdiv_(d_p, average);
      p.add_(momentum, -options.learning_rate_);
    } else {
      p.addcdiv_(d_p, average, -options.learning_rate_);
    }
  }
}
} // namespace optim
} // namespace torch
