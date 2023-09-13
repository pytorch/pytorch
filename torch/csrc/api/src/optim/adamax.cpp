#include <torch/optim/Adamax.h>

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

AdamaxOptions::AdamaxOptions(double lr) : lr_(lr) {}

bool operator==(const AdamaxOptions& lhs, const AdamaxOptions& rhs) {
  return (lhs.lr() == rhs.lr()) &&
      (std::get<0>(lhs.betas()) == std::get<0>(rhs.betas())) &&
      (std::get<1>(lhs.betas()) == std::get<1>(rhs.betas())) &&
      (lhs.eps() == rhs.eps()) && (lhs.weight_decay() == rhs.weight_decay());
}

void AdamaxOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
}

void AdamaxOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
}

double AdamaxOptions::get_lr() const {
  return lr();
}

void AdamaxOptions::set_lr(const double lr) {
  this->lr(lr);
}

bool operator==(const AdamaxParamState& lhs, const AdamaxParamState& rhs) {
  return (lhs.step() == rhs.step()) &&
      torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
      torch::equal(lhs.inf_norm(), rhs.inf_norm());
}

void AdamaxParamState::serialize(
    torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(inf_norm);
}

void AdamaxParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, inf_norm);
}

Tensor Adamax::step(LossClosure closure) {
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
      auto param_state =
          state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
      auto& options = static_cast<AdamaxOptions&>(group.options());

      // State initialization
      if (param_state == state_.end()) {
        auto state = std::make_unique<AdamaxParamState>();
        state->step(0);
        // Exponential moving average of gradient values
        state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
        // Infinity norm
        state->inf_norm(torch::zeros_like(p, MemoryFormat::Preserve));

        state_[c10::guts::to_string(p.unsafeGetTensorImpl())] =
            std::move(state);
      }

      auto& state = static_cast<AdamaxParamState&>(
          *state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
      auto& exp_avg = state.exp_avg();
      auto& inf_norm = state.inf_norm();

      state.step(state.step() + 1);
      auto beta1 = std::get<0>(options.betas());
      auto beta2 = std::get<1>(options.betas());

      auto bias_correction1 = 1 - std::pow(beta1, state.step());

      if (options.weight_decay() != 0) {
        grad = grad.add(p, options.weight_decay());
      }

      exp_avg.mul_(beta1).add_(grad, 1 - beta1);
      // inf_norm = max(b2 * inf_norm, abs(gt) + eps)
      torch::max_out(
          inf_norm, inf_norm.mul(beta2), grad.abs().add(options.eps()));

      Tensor num = options.lr() * exp_avg;
      Tensor denom = inf_norm * bias_correction1;

      p.addcdiv_(num, denom, -1);
    }
  }
  return loss;
}

void Adamax::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Adamax::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch
