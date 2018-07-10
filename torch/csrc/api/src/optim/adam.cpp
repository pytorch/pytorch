#include <torch/optim/adam.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>

namespace torch {
namespace optim {

AdamOptions::AdamOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

const AdamOptions& Adam::options() const noexcept {
  return options_;
}

void Adam::step() {
  for (size_t i = 0; i < parameters_.size(); ++i) {
    auto& grad = parameters_.at(i).grad();
    auto& p = parameters_.at(i).data();
    if (!grad.defined()) {
      continue;
    }

    auto exp_average = buffer_at(exp_average_buffers_, i).data();
    auto exp_average_sq = buffer_at(exp_average_sq_buffers_, i).data();

    step_buffers_.at(i) += 1;

    auto d_p = torch::autograd::as_variable_ref(grad).data();
    if (options_.weight_decay_ > 0) {
      d_p.add_(p, options_.weight_decay_);
    }

    exp_average.mul_(options_.beta1_).add_(d_p, 1 - options_.beta1_);
    exp_average_sq.mul_(options_.beta2_)
        .addcmul_(d_p, d_p, 1 - options_.beta2_);

    at::Tensor denom;
    if (options_.amsgrad_) {
      auto max_exp_average_sq =
          buffer_at(max_exp_average_sq_buffers_, i).data();
      torch::max_out(max_exp_average_sq, max_exp_average_sq, exp_average_sq);
      denom = max_exp_average_sq.sqrt().add_(options_.eps_);
    } else {
      denom = exp_average_sq.sqrt().add_(options_.eps_);
    }

    const auto bias_correction1 =
        1 - std::pow(options_.beta1_, step_buffers_.at(i));
    const auto bias_correction2 =
        1 - std::pow(options_.beta2_, step_buffers_.at(i));
    const auto step_size = options_.learning_rate_ *
        std::sqrt(bias_correction2) / bias_correction1;

    p.addcdiv_(exp_average, denom, -step_size);
  }
}

} // namespace optim
} // namespace torch
