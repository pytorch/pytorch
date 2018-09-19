#include <torch/optim/adam.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>

namespace torch {
namespace optim {

AdamOptions::AdamOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

void Adam::step() {
  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor p = parameters_.at(i);
    if (!p.grad().defined()) {
      continue;
    }

    if (options.weight_decay_ > 0) {
      p.grad() = p.grad() + options.weight_decay_ * p;
    }

    auto& exp_average = buffer_at(exp_average_buffers_, i);
    auto& exp_average_sq = buffer_at(exp_average_sq_buffers_, i);

    buffer_at(step_buffers_, i) += 1;

    exp_average.mul_(options.beta1_).add_(p.grad(), 1 - options.beta1_);
    exp_average_sq.mul_(options.beta2_)
        .addcmul_(p.grad(), p.grad(), 1 - options.beta2_);

    Tensor denom = exp_average_sq;
    if (options.amsgrad_) {
      auto& max_exp_average_sq = buffer_at(max_exp_average_sq_buffers_, i);
      max_exp_average_sq = torch::max(max_exp_average_sq, exp_average_sq);
      denom = max_exp_average_sq;
    }

    const auto bias_correction1 =
        1 - std::pow(options.beta1_, buffer_at(step_buffers_, i));
    const auto bias_correction2 =
        1 - std::pow(options.beta2_, buffer_at(step_buffers_, i));
    const auto step_size =
        options.learning_rate_ * std::sqrt(bias_correction2) / bias_correction1;

    NoGradGuard guard;
    p.addcdiv_(exp_average, denom.sqrt() + options.eps_, -step_size);
  }
}

} // namespace optim
} // namespace torch
