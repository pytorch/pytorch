#include <torch/optim/adam.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>

namespace torch {
namespace optim {
AdamOptions::AdamOptions(double lr) : lr_(lr) {}

bool operator==(const AdamOptions& lhs, const AdamOptions& rhs) {
  return (lhs.lr() == rhs.lr()) &&
         (lhs.beta1() == rhs.beta1()) &&
         (lhs.beta2() == rhs.beta2()) &&
         (lhs.eps() == rhs.eps()) &&
         (lhs.weight_decay() == rhs.weight_decay() &&
         (lhs.amsgrad() == rhs.amsgrad()));
}

void AdamOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta1);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(beta2);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(amsgrad);
}

void AdamOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta1);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, beta2);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, amsgrad);
}

bool operator==(const AdamParamState& lhs, const AdamParamState& rhs) {
  return (lhs.step() == rhs.step()) &&
          torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
          torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq()) &&
          torch::equal(lhs.max_exp_avg_sq(), rhs.max_exp_avg_sq());
}

void AdamParamState::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_exp_avg_sq);
}

void AdamParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg_sq);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, max_exp_avg_sq);
}

void Adam::step() {
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }
      auto grad = p.grad().data();
      TORCH_CHECK(!grad.is_sparse(), "Adam does not support sparse gradients"/*, please consider SparseAdam instead*/);
      auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
      auto& options = static_cast<AdamOptions&>(group.options());
      // State initialization
      if(param_state == state_.end()) {
        auto state = std::make_unique<AdamParamState>();
        state->step(0);
        state->exp_avg(torch::zeros_like(p.data(), MemoryFormat::Preserve));
        state->exp_avg_sq(torch::zeros_like(p.data(), MemoryFormat::Preserve));
        if(options.amsgrad()) {
          state->max_exp_avg_sq(torch::zeros_like(p.data(), MemoryFormat::Preserve));
        }
        state_[c10::guts::to_string(p.unsafeGetTensorImpl())] = std::move(state);
      }
      auto state = static_cast<AdamParamState&>(*state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
      // Exponential moving average of gradient values
      auto exp_avg = state.exp_avg();
      // Exponential moving average of squared gradient values
      auto exp_avg_sq = state.exp_avg_sq();
      Tensor max_exp_avg_sq;
      if(options.amsgrad()) {
        // Maintains max of all exp. moving avg. of sq. grad. values
        max_exp_avg_sq = state.max_exp_avg_sq();
      }
      state.step(state.step()+1);
      auto bias_correction1 = 1 - std::pow(options.beta1(), state.step());
      auto bias_correction2 = 1 - std::pow(options.beta2(), state.step());

      if(options.weight_decay() != 0) {
        grad = grad.add(p.data(), options.weight_decay());
      }

      // Decay the first and second moment running average coefficient
      exp_avg.mul_(options.beta1()).add_(grad, 1 - options.beta1());
      exp_avg_sq.mul_(options.beta2()).addcmul_(grad, grad, 1 - options.beta2());

      Tensor denom;
      if(options.amsgrad()) {
        // Maintains the maximum of all 2nd moment running avg. till now
        at::max_out(max_exp_avg_sq, exp_avg_sq, max_exp_avg_sq);
        // Use the max. for normalizing running avg. of gradient
        denom = (max_exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(options.eps());
      } else {
        denom = (exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(options.eps());
      }

      auto step_size = options.lr() / bias_correction1;
      p.data().addcdiv_(exp_avg, denom, -step_size);
    }
  }
}

void Adam::add_parameters(const std::vector<Tensor>& parameters) {
  param_groups_.emplace_back(OptimizerParamGroup(parameters, defaults_->clone()));
}

const std::vector<Tensor>& Adam::parameters() const noexcept {
  return param_groups_.at(0).params();
}

std::vector<Tensor>& Adam::parameters() noexcept {
  return param_groups_.at(0).params();
}

size_t Adam::size() const noexcept {
  return _size_new_design();
}

void Adam::save(serialize::OutputArchive& archive) const {
  // serialize(*this, archive);
}

void Adam::load(serialize::InputArchive& archive) {
  // serialize(*this, archive);
}
} // namespace optim
} // namespace torch
