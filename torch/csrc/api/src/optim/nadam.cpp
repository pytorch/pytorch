#include <torch/optim/nadam.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

#include <cmath>
#include <functional>

namespace torch {
namespace optim {

NAdamOptions::NAdamOptions(double lr) : lr_(lr) {}

bool operator==(const NAdamOptions& lhs, const NAdamOptions& rhs) {
  return (lhs.lr() == rhs.lr()) &&
      (std::get<0>(lhs.betas()) == std::get<0>(rhs.betas())) &&
      (std::get<1>(lhs.betas()) == std::get<1>(rhs.betas())) &&
      (lhs.eps() == rhs.eps()) &&
      (lhs.weight_decay() == rhs.weight_decay() &&
       (lhs.momentum_decay() == rhs.momentum_decay()));
}

void NAdamOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum_decay);
}

void NAdamOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, momentum_decay);
}

double NAdamOptions::get_lr() const {
  return lr();
}

void NAdamOptions::set_lr(const double lr) {
  this->lr(lr);
}

bool operator==(const NAdamParamState& lhs, const NAdamParamState& rhs) {
  return (lhs.step() == rhs.step()) &&
      torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
      torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq()) &&
      (lhs.mu_product() == rhs.mu_product());
}

void NAdamParamState::serialize(
    torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(mu_product);
}

void NAdamParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg_sq);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, mu_product);
}

Tensor NAdam::step(LossClosure closure) {
  NoGradGuard no_grad;
  Tensor loss = {};
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }
      auto grad = p.grad();
      TORCH_CHECK(!grad.is_sparse(), "NAdam does not support sparse gradients" /*, please consider SparseAdam instead*/);
      auto param_state = state_.find(p.unsafeGetTensorImpl());
      auto& options = static_cast<NAdamOptions&>(group.options());

      // State initialization
      if (param_state == state_.end()) {
        auto state = std::make_unique<NAdamParamState>();
        state->step(0);
        state->mu_product(1.);
        // Exponential moving average of gradient values
        state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
        // Exponential moving average of squared gradient values
        state->exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
        state_[p.unsafeGetTensorImpl()] = std::move(state);
      }

      auto& state =
          static_cast<NAdamParamState&>(*state_[p.unsafeGetTensorImpl()]);
      auto& exp_avg = state.exp_avg();
      auto& exp_avg_sq = state.exp_avg_sq();
      auto lr = options.lr();

      state.step(state.step() + 1);
      auto beta1 = std::get<0>(options.betas());
      auto beta2 = std::get<1>(options.betas());

      auto bias_correction2 = 1 - std::pow(beta2, state.step());

      if (options.weight_decay() != 0) {
        grad = grad.add(p, options.weight_decay());
      }

      // calculate the momentum cache \mu^{t} and \mu^{t+1}
      auto mu = beta1 * (1. - 0.5 * (pow(0.96, state.step() * options.momentum_decay())));
      auto mu_next = beta1 * (1. - 0.5 * (pow(0.96, (state.step() + 1) * options.momentum_decay())));

      // update mu_product
      state.mu_product(state.mu_product() * mu);

      // Decay the first and second moment running average coefficient
      exp_avg.lerp_(grad, 1 - beta1);
      exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);
      Tensor denom = (exp_avg_sq / bias_correction2).sqrt().add(options.eps());

      auto mu_product_next = state.mu_product() * mu_next;
      p.addcdiv_(
        grad, denom, (-lr * (1. - mu) / (1. - state.mu_product())));
      p.addcdiv_(
          exp_avg, denom, (-lr * mu_next) / (1. - mu_product_next));
    }
  }
  return loss;
}

void NAdam::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void NAdam::load(serialize::InputArchive& archive) {
  IValue pytorch_version;
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch
