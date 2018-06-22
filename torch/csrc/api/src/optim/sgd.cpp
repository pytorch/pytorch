#include <torch/optim/sgd.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {

SGDOptions::SGDOptions(double learning_rate) : learning_rate_(learning_rate) {}

SGD::SGD(std::shared_ptr<nn::Module> model, const SGDOptions& options)
    : Optimizer(model), options_(options) {}

const SGDOptions& SGD::options() const noexcept {
  return options_;
}

at::Scalar SGD::step(std::function<at::Scalar()> closure) {
  at::Scalar loss = closure();
  for (auto& parameter : model_->parameters()) {
    auto& name = parameter.key;
    auto& grad = parameter->grad();
    auto& p = parameter->data();
    if (!grad.defined()) {
      continue;
    }

    auto d_p = torch::autograd::as_variable_ref(grad).data();
    if (options_.weight_decay_ > 0) {
      d_p.add_(p, options_.weight_decay_);
    }

    if (options_.momentum_ != 0) {
      at::Tensor buf;
      if (momentum_buffers_.find(name) == momentum_buffers_.end()) {
        buf = momentum_buffers_[name] = at::zeros_like(p);
        buf.mul_(options_.momentum_).add_(d_p);
      } else {
        buf = momentum_buffers_[name];
        buf.mul_(options_.momentum_).add_(d_p, 1 - options_.dampening_);
      }

      if (options_.nesterov_) {
        d_p = d_p.add(buf, options_.momentum_);
      } else {
        d_p = buf;
      }
    }

    p.add_(d_p, -options_.learning_rate_);
  }
  return loss;
}
} // namespace optim
} // namespace torch
