#include <torch/optim/rmsprop.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/serialize/archive.h>
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

    if (options.weight_decay() > 0) {
      NoGradGuard guard;
      p.grad() = p.grad() + options.weight_decay() * p;
    }

    auto square_average = buffer_at(square_average_buffers, i);
    square_average.mul_(options.alpha())
        .addcmul_(p.grad(), p.grad(), 1.0 - options.alpha());

    Tensor average;
    if (options.centered() > 0) {
      auto& grad_average = buffer_at(grad_average_buffers, i);
      grad_average.mul_(options.alpha()).add_(p.grad(), 1.0 - options.alpha());
      average = square_average.addcmul(grad_average, grad_average, -1.0)
                    .sqrt()
                    .add_(options.eps());
    } else {
      average = square_average.sqrt().add_(options.eps());
    }

    NoGradGuard guard;
    if (options.momentum() > 0) {
      auto& momentum = buffer_at(momentum_buffers, i);
      momentum.mul_(options.momentum()).addcdiv_(p.grad(), average);
      p.add_(momentum, -options.learning_rate());
    } else {
      p.addcdiv_(p.grad(), average, -options.learning_rate());
    }
  }
}

void RMSprop::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void RMSprop::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch
