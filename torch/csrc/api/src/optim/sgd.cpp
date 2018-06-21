#include <torch/optim/sgd.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {
at::Scalar SGD::step(std::function<at::Scalar()> closure) {
  at::Scalar loss = closure();
  for (auto& parameter : model_->parameters()) {
    auto& name = parameter.key;
    auto& grad = parameter->grad();
    auto& p = parameter->data();
    if (!grad.defined())
      continue;

    auto d_p = torch::autograd::as_variable_ref(grad).data();
    if (weight_decay_ > 0) {
      d_p.add_(p, weight_decay_);
    };

    if (momentum_ != 0) {
      at::Tensor buf;
      if (momentum_buffers_.find(name) == momentum_buffers_.end()) {
        buf = momentum_buffers_[name] = at::zeros_like(p);
        buf.mul_(momentum_).add_(d_p);
      } else {
        buf = momentum_buffers_[name];
        buf.mul_(momentum_).add_(d_p, 1 - dampening_);
      }

      if (nesterov_) {
        d_p = d_p.add(buf, momentum_);
      } else {
        d_p = buf;
      }
    }

    p.add_(d_p, -lr_);
  }
  return loss;
}

void SGD::init_state() {
  momentum_buffers_.clear();
}
} // namespace optim
} // namespace torch
