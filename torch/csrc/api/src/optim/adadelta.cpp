#include <torch/optim/adadelta.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

#include <functional>

#include <torch/types.h>

namespace torch {
namespace optim {

AdadeltaOptions::AdadeltaOptions(double lr) : lr_(lr) {}

bool operator==(const AdadeltaOptions& lhs, const AdadeltaOptions& rhs) {
  return (lhs.lr() == rhs.lr()) && (lhs.rho() == rhs.rho()) &&
      (lhs.weight_decay() == rhs.weight_decay()) &&
      (lhs.eps() == rhs.eps());
}

void AdadeltaOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(rho);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
}

void AdadeltaOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, rho);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
}

double AdadeltaOptions::get_lr() const {
  return lr();
}

void AdadeltaOptions::set_lr(const double lr) {
  this->lr(lr);
}

bool operator==(const AdadeltaParamState& lhs, const AdadeltaParamState& rhs) {
  return torch::equal(lhs.square_avg(), rhs.square_avg()) &&
    torch::equal(lhs.accumulate(), rhs.accumulate());
}

void AdadeltaParamState::serialize(
    torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(square_avg);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(accumulate);
}

void AdadeltaParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, square_avg);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, accumulate);
}


/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/adadelta.py
Tensor Adadelta::step(LossClosure closure) {
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
      auto grad = p.grad();;

      auto param_state =
        state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));

      // State initialization
      if (param_state == state_.end()) {
        auto state = std::make_unique<AdadeltaParamState>();
        state->square_avg(torch::full_like(
            p.data(), 0, at::MemoryFormat::Preserve
          ));
        state->accumulate(torch::full_like(
            p.data(), 0, at::MemoryFormat::Preserve
          ));
        state_[c10::guts::to_string(p.unsafeGetTensorImpl())] =
            std::move(state);
      }

      auto& state = static_cast<AdadeltaParamState&>(
          *state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);

      auto& options = static_cast<AdadeltaOptions&>(group.options());

      state.square_avg(
        state.square_avg().mul_(options.rho()).addcmul_(grad, grad, 1 - options.rho())
      );
      auto std = state.square_avg().add(options.eps()).sqrt_();
      auto delta = state.accumulate().add(options.eps()).sqrt_();
      delta.div_(std).mul_(grad);
      state.accumulate(
        state.accumulate().mul_(options.rho()).addcmul_(delta, delta, 1 - options.rho())
      );
      p.add_(delta, -options.lr());
    }
  }
  return loss;
}

void Adadelta::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void Adadelta::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
  std::cout << "gel";
}
} // namespace optim
} // namespace torch
