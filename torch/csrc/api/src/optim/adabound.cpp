#include <include/adabound.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>

namespace torch {
namespace optim {

AdaBoundOptions::AdaBoundOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

void AdaBound::step() {
  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor p = parameters_.at(i);
    if (!p.grad().defined()) {
      continue;
    }
    if( !options.decoupled_weight_decay_ && options.weight_decay_ > 0) {
      p.grad() = p.grad() + options.weight_decay_ * p;
    }

    auto& exp_average = buffer_at(exp_average_buffers, i);
    auto& exp_average_sq = buffer_at(exp_average_sq_buffers, i);

    buffer_at(step_buffers, i) += 1;
    const uint64_t timestep = buffer_at(step_buffers, i);

    exp_average.mul_(options.beta1_).add_(p.grad(), 1 - options.beta1_);
    exp_average_sq.mul_(options.beta2_)
        .addcmul_(p.grad(), p.grad(), 1 - options.beta2_);

    Tensor denom = exp_average_sq;
    if (options.amsgrad_) {
      auto& max_exp_average_sq = buffer_at(max_exp_average_sq_buffers, i);
      max_exp_average_sq = torch::max(max_exp_average_sq, exp_average_sq);
      denom = max_exp_average_sq;
      denom.sqrt_().add_(options.eps_);
    }

    const auto bias_correction1 =
        1. - std::pow(options.beta1_, buffer_at(step_buffers, i));
    const auto bias_correction2 =
        1. - std::pow(options.beta2_, buffer_at(step_buffers, i));
    const auto step_size =
        options.learning_rate_ * std::sqrt(bias_correction2) / bias_correction1;
    const auto lower_bound = options.final_lr_ * (1. - 1. / (options.gamma_ * timestep + 1.));
    const auto upper_bound = options.final_lr_ * (1. + 1. / (options.gamma_ * timestep));

    NoGradGuard guard;
    auto step_tensor = torch::full_like(denom, step_size);
    step_tensor.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_average);

    if(options.decoupled_weight_decay_ && options.weight_decay_ > 0) {
        step_tensor += -(options.weight_decay_ * step_size) * p;
    }
    p.add_(step_tensor);
  }
}

void AdaBound::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void AdaBound::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch
