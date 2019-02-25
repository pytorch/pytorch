#include <include/adamw.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>

namespace torch {
namespace optim {

AdamWOptions::AdamWOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

void AdamW::step() {
  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor p = parameters_.at(i);
    if (!p.grad().defined()) {
      continue;
    }


    auto& exp_average = buffer_at(exp_average_buffers, i);
    auto& exp_average_sq = buffer_at(exp_average_sq_buffers, i);

    buffer_at(step_buffers, i) += 1;

    exp_average.mul_(options.beta1_).add_(p.grad(), 1 - options.beta1_);
    exp_average_sq.mul_(options.beta2_)
        .addcmul_(p.grad(), p.grad(), 1 - options.beta2_);

    Tensor denom = exp_average_sq;
    if (options.amsgrad_) {
      auto& max_exp_average_sq = buffer_at(max_exp_average_sq_buffers, i);
      max_exp_average_sq = torch::max(max_exp_average_sq, exp_average_sq);
      denom = max_exp_average_sq;
    }

    const auto bias_correction1 =
        1 - std::pow(options.beta1_, buffer_at(step_buffers, i));
    const auto bias_correction2 =
        1 - std::pow(options.beta2_, buffer_at(step_buffers, i));
    const auto step_size =
        options.learning_rate_ * std::sqrt(bias_correction2) / bias_correction1;

    NoGradGuard guard;

    if(options.weight_decay_ > 0) {
        Tensor decoupled_weight_decay = p * (-options.weight_decay_ * step_size);
        p.addcdiv_(exp_average, denom.sqrt() + options.eps_, -step_size);
        p.add_(decoupled_weight_decay);
    } else {
        p.addcdiv_(exp_average, denom.sqrt() + options.eps_, -step_size);
    }
  }
}

void AdamW::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void AdamW::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch
