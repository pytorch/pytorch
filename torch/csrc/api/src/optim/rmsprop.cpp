#include <torch/optim/rmsprop.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {

RMSpropOptions::RMSpropOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

const RMSpropOptions& RMSprop::options() const noexcept {
  return options_;
}

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
    if (options_.weight_decay_ > 0) {
      d_p.add_(p, options_.weight_decay_);
    }

    auto square_average = buffer_at(square_average_buffers_, i).data();
    square_average.mul_(options_.alpha_)
        .addcmul_(d_p, d_p, 1.0 - options_.alpha_);

    at::Tensor average;
    if (options_.centered_ > 0) {
      auto grad_average = buffer_at(grad_average_buffers_, i).data();
      grad_average.mul_(options_.alpha_).add_(d_p, 1.0 - options_.alpha_);
      average = square_average.addcmul(grad_average, grad_average, -1.0)
                    .sqrt()
                    .add_(options_.eps_);
    } else {
      average = square_average.sqrt().add_(options_.eps_);
    }

    if (options_.momentum_ > 0) {
      auto momentum = buffer_at(momentum_buffers_, i).data();
      momentum.mul_(options_.momentum_).addcdiv_(d_p, average);
      p.add_(momentum, -options_.learning_rate_);
    } else {
      p.addcdiv_(d_p, average, -options_.learning_rate_);
    }
  }
}
} // namespace optim
} // namespace torch
