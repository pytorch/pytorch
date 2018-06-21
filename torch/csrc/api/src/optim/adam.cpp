#include <torch/optim/adam.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>

namespace torch {
namespace optim {

Adam::Adam(std::shared_ptr<nn::Module> model, double lr)
    : Optimizer(model), lr_(lr) {}

at::Scalar Adam::step(std::function<at::Scalar()> closure) {
  at::Scalar loss = closure();
  for (auto& parameter : model_->parameters()) {
    auto& name = parameter.key;
    auto& grad = parameter->grad();
    auto& p = parameter->data();
    if (!grad.defined())
      continue;

    if (step_buffer_.find(name) == step_buffer_.end()) {
      step_buffer_[name] = 0;
      exp_avg_buffer_[name] = at::zeros_like(p);
      exp_avg_sq_buffer_[name] = at::zeros_like(p);
      if (amsgrad_) {
        max_exp_avg_sq_buffer_[name] = at::zeros_like(p);
      };
    }

    auto& step = step_buffer_[name];
    auto& exp_avg = exp_avg_buffer_[name];
    auto& exp_avg_sq = exp_avg_sq_buffer_[name];

    step += 1;

    auto d_p = torch::autograd::as_variable_ref(grad).data();
    if (weight_decay_ > 0) {
      d_p.add_(p, weight_decay_);
    }

    exp_avg.mul_(beta1_).add_(d_p, 1 - beta1_);
    exp_avg_sq.mul_(beta2_).addcmul_(d_p, d_p, 1 - beta2_);

    at::Tensor denom;
    if (amsgrad_) {
      auto& max_exp_avg_sq = max_exp_avg_sq_buffer_[name];
      at::max_out(max_exp_avg_sq, max_exp_avg_sq, exp_avg_sq);
      denom = max_exp_avg_sq.sqrt().add_(eps_);
    } else {
      denom = exp_avg_sq.sqrt().add_(eps_);
    };

    auto bias_correction1 = 1 - std::pow(beta1_, step);
    auto bias_correction2 = 1 - std::pow(beta2_, step);
    auto step_size = lr_ * std::sqrt(bias_correction2) / bias_correction1;

    p.addcdiv_(exp_avg, denom, -step_size);
  }
  return loss;
}

} // namespace optim
} // namespace torch
