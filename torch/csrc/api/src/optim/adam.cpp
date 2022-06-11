#include <torch/optim/adam.h>

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

AdamOptions::AdamOptions(double lr) : lr_(lr) {}

bool operator==(const AdamOptions& lhs, const AdamOptions& rhs) {
  return (lhs.lr() == rhs.lr()) &&
      (std::get<0>(lhs.betas()) == std::get<0>(rhs.betas())) &&
      (std::get<1>(lhs.betas()) == std::get<1>(rhs.betas())) &&
      (lhs.eps() == rhs.eps()) &&
      (lhs.weight_decay() == rhs.weight_decay() &&
       (lhs.amsgrad() == rhs.amsgrad()));
}

void AdamOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(amsgrad);
}

void AdamOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, amsgrad);
}

double AdamOptions::get_lr() const {
  return lr();
}

void AdamOptions::set_lr(const double lr) {
  this->lr(lr);
}

bool operator==(const AdamParamState& lhs, const AdamParamState& rhs) {
  return (lhs.step() == rhs.step()) &&
      torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
      torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq()) &&
      torch::equal_if_defined(lhs.max_exp_avg_sq(), rhs.max_exp_avg_sq());
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

Tensor Adam::step(LossClosure closure) {
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
      TORCH_CHECK(!grad.is_sparse(), "Adam does not support sparse gradients" /*, please consider SparseAdam instead*/);
      auto param_state =
          state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
      auto& options = static_cast<AdamOptions&>(group.options());

      // State initialization
      if (param_state == state_.end()) {
        auto state = std::make_unique<AdamParamState>();
        state->step(0);
        // Exponential moving average of gradient values
        state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
        // Exponential moving average of squared gradient values
        state->exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
        if (options.amsgrad()) {
          // Maintains max of all exp. moving avg. of sq. grad. values
          state->max_exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
        }
        state_[c10::guts::to_string(p.unsafeGetTensorImpl())] =
            std::move(state);
      }

      auto& state = static_cast<AdamParamState&>(
          *state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
      auto& exp_avg = state.exp_avg();
      auto& exp_avg_sq = state.exp_avg_sq();
      auto& max_exp_avg_sq = state.max_exp_avg_sq();

      state.step(state.step() + 1);
      auto beta1 = std::get<0>(options.betas());
      auto beta2 = std::get<1>(options.betas());

      auto bias_correction1 = 1 - std::pow(beta1, state.step());
      auto bias_correction2 = 1 - std::pow(beta2, state.step());

      if (options.weight_decay() != 0) {
        grad = grad.add(p, options.weight_decay());
      }

      // Decay the first and second moment running average coefficient
      exp_avg.mul_(beta1).add_(grad, 1 - beta1);
      exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

      Tensor denom;
      if (options.amsgrad()) {
        // Maintains the maximum of all 2nd moment running avg. till now
        torch::max_out(max_exp_avg_sq, exp_avg_sq, max_exp_avg_sq);
        // Use the max. for normalizing running avg. of gradient
        denom = (max_exp_avg_sq.sqrt() / sqrt(bias_correction2))
                    .add_(options.eps());
      } else {
        denom =
            (exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(options.eps());
      }

      auto step_size = options.lr() / bias_correction1;
      p.addcdiv_(exp_avg, denom, -step_size);
    }
  }
  return loss;
}

void Adam::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Adam::load(serialize::InputArchive& archive) {
  IValue pytorch_version;
  if (archive.try_read("pytorch_version", pytorch_version)) {
    serialize(*this, archive);
  } else { // deserializing archives saved in old format (prior to
           // version 1.5.0)
    TORCH_WARN(
        "Your serialized Adam optimizer is still using the old serialization format. "
        "You should re-save your Adam optimizer to use the new serialization format.");
    std::vector<int64_t> step_buffers;
    std::vector<at::Tensor> exp_average_buffers;
    std::vector<at::Tensor> exp_average_sq_buffers;
    std::vector<at::Tensor> max_exp_average_sq_buffers;
    torch::optim::serialize(archive, "step_buffers", step_buffers);
    torch::optim::serialize(
        archive, "exp_average_buffers", exp_average_buffers);
    torch::optim::serialize(
        archive, "exp_average_sq_buffers", exp_average_sq_buffers);
    torch::optim::serialize(
        archive, "max_exp_average_sq_buffers", max_exp_average_sq_buffers);
    // since there were no param_groups prior to version 1.5.0, assuming all
    // tensors are now in one param_group
    std::vector<Tensor> params = param_groups_.at(0).params();
    for (const auto idx : c10::irange(step_buffers.size())) {
      auto state = std::make_unique<AdamParamState>();
      state->step(step_buffers.at(idx));
      state->exp_avg(exp_average_buffers.at(idx));
      state->exp_avg_sq(exp_average_sq_buffers.at(idx));
      if (idx < max_exp_average_sq_buffers.size()) {
        state->max_exp_avg_sq(max_exp_average_sq_buffers.at(idx));
      }
      state_[c10::guts::to_string(params.at(idx).unsafeGetTensorImpl())] =
          std::move(state);
    }
  }
}
} // namespace optim
} // namespace torch
