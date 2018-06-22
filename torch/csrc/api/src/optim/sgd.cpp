#include <torch/optim/sgd.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {
SGDOptions::SGDOptions(double learning_rate) : learning_rate_(learning_rate) {}

const SGDOptions& SGD::options() const noexcept {
  return options_;
}

void SGD::step() {
  for (size_t i = 0; i < parameters_.size(); ++i) {
    auto& grad = parameters_[i].grad();
    auto& p = parameters_[i].data();

    if (!grad.defined()) {
      continue;
    }

    auto d_p = torch::autograd::as_variable_ref(grad).data();
    if (options_.weight_decay_ > 0) {
      d_p.add_(p, options_.weight_decay_);
    }

    if (options_.momentum_ != 0) {
      AT_ASSERT(momentum_buffers_.size() == parameters_.size());
      auto momentum = momentum_buffers_[i];

      if (iteration_ == 0) {
        momentum.data().mul_(options_.momentum_).add_(d_p);
      } else {
        momentum.data()
            .mul_(options_.momentum_)
            .add_(d_p, 1 - options_.dampening_);
      }

      if (options_.nesterov_) {
        d_p = d_p.add(momentum.data(), options_.momentum_);
      } else {
        d_p = momentum;
      }
    }

    p.add_(d_p, -options_.learning_rate_);
  }
  iteration_ += 1;
}
} // namespace optim
} // namespace torch
