#include <torch/optim/adagrad.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {

AdagradOptions::AdagradOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
void Adagrad::step() {
  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor p = parameters_.at(i);
    if (!p.grad().defined()) {
      continue;
    }

    if (options.weight_decay_ > 0) {
      NoGradGuard guard;
      p.grad() = p.grad() + options.weight_decay_ * p;
    }

    buffer_at(step_buffers, i) += 1.0;
    const auto clr = options.learning_rate_ /
        (1.0 + (buffer_at(step_buffers, i) - 1.0) * options.lr_decay_);

    auto& sum = buffer_at(sum_buffers, i);
    sum.addcmul_(p.grad(), p.grad(), 1.0);
    const auto std = buffer_at(sum_buffers, i).sqrt().add_(1e-10);

    NoGradGuard guard;
    p.addcdiv_(p.grad(), std, -clr);
  }
}

void Adagrad::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Adagrad::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch
