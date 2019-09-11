#include <torch/optim/sgd.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/pimpl.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {
SGDOptions::SGDOptions(double learning_rate) : learning_rate_(learning_rate) {}

void SGD::step() {
  for (size_t i = 0; i < parameters_.size(); ++i) {
    Tensor p = parameters_.at(i);

    if (!p.grad().defined()) {
      continue;
    }

    auto update = p.grad();

    if (options.weight_decay_ > 0) {
      NoGradGuard guard;
      update += options.weight_decay_ * p;
    }

    if (options.momentum_ != 0) {
      const auto dampening = iteration_ == 0 ? 1 : 1 - options.dampening_;
      auto& momentum = buffer_at(momentum_buffers, i);
      momentum = (options.momentum_ * momentum) + (dampening * update);
      if (options.nesterov_) {
        // See github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617
        // for notes on this implementation of nesterov momentum.
        update += options.momentum_ * momentum;
      } else {
        update = momentum;
      }
    }

    NoGradGuard guard;
    p.add_(-options.learning_rate_ * update);
  }
  iteration_ += 1;
}

void SGD::save(serialize::OutputArchive& archive) const {
  optim::serialize(archive, "momentum_buffers", momentum_buffers);
}

void SGD::load(serialize::InputArchive& archive) {
  optim::serialize(archive, "momentum_buffers", momentum_buffers);
}
} // namespace optim
} // namespace torch
