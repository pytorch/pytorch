#include <torch/optim/rmsprop.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace optim {

RMSpropOptions::RMSpropOptions(double lr)
    : lr_(lr) {}

bool operator==(const RMSpropOptions& lhs, const RMSpropOptions& rhs) {
  return (lhs.lr() == rhs.lr()) &&
          (lhs.alpha() == rhs.alpha()) &&
          (lhs.eps() == rhs.eps()) &&
          (lhs.weight_decay() == rhs.weight_decay()) &&
          (lhs.momentum() == rhs.momentum()) &&
          (lhs.centered() == rhs.centered());
}

void RMSpropOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(alpha);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(centered);
}

void RMSpropOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, alpha);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, momentum);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, centered);
}

bool operator==(const RMSpropParamState& lhs, const RMSpropParamState& rhs) {
  return (lhs.step() == rhs.step()) &&
         torch::equal(lhs.momentum_buffer(), rhs.momentum_buffer()) &&
         torch::equal(lhs.square_avg(), rhs.square_avg()) &&
         torch::equal(lhs.grad_avg(), rhs.grad_avg());
}

void RMSpropParamState::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum_buffer);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(square_avg);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(grad_avg);
}

void RMSpropParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, momentum_buffer);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, square_avg);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, grad_avg);
}

/// Adapted from
/// https://github.com/pytorch/pytorch/blob/master/torch/optim/rmsprop.py
//check where Nograd should be used
void RMSprop::step() {
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }
      auto grad = p.grad().data();
      TORCH_CHECK(!grad.is_sparse(), "RMSprop does not support sparse gradients");
      auto param_state = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
      auto& options = static_cast<RMSpropOptions&>(group.options());

      if(param_state == state_.end()) {
        auto state = std::make_unique<RMSpropParamState>();
        state->step(0);
        state->square_avg(torch::zeros_like(p.data(), MemoryFormat::Preserve));
        // TODO: think and test if I need to initialize momentum_buffer with an empty tensor when momentum is neg
        if(options.momentum() > 0) {
          state->momentum_buffer(torch::zeros_like(p.data(), MemoryFormat::Preserve));
        }
        // TODO: think and test if I need to initialize grad_avg with an empty tensor when centered is False
        if(options.centered()) {
          state->grad_avg(torch::zeros_like(p.data(), MemoryFormat::Preserve));
        }
        state_[c10::guts::to_string(p.unsafeGetTensorImpl())] = std::move(state);
      }

      auto state = static_cast<RMSpropParamState&>(*state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
      auto square_avg = state.square_avg();
      auto alpha = options.alpha();

      state.step()+=1;

      if (options.weight_decay() != 0) {
        grad = grad.add(p.data(), options.weight_decay());
      }

      square_avg.mul_(alpha).addcmul_(grad, grad, 1 - alpha);

      Tensor avg;
      if(options.centered()) {
        auto grad_avg = state.grad_avg();
        grad_avg.mul_(alpha).add_(grad, 1-alpha);
        avg = square_avg.addcmul(grad_avg, grad_avg, -1).sqrt_().add_(options.eps());
      } else {
        avg = square_avg.sqrt().add_(options.eps());
      }

      if(options.momentum() > 0) {
        auto buf = state.momentum_buffer();
        buf.mul_(options.momentum()).addcdiv_(grad, avg);
        p.data().add_(buf, -options.lr());
      } else {
        p.data().addcdiv_(grad, avg, -options.lr());
      }
    }
  }
}

void RMSprop::save(serialize::OutputArchive& archive) const {
  //serialize(*this, archive);
}

void RMSprop::load(serialize::InputArchive& archive) {
  //serialize(*this, archive);
}

void RMSprop::add_parameters(const std::vector<Tensor>& parameters) {
  param_groups_.emplace_back(OptimizerParamGroup(parameters, defaults_->clone()));
}

const std::vector<Tensor>& RMSprop::parameters() const noexcept {
  return param_groups_.at(0).params();
}

std::vector<Tensor>& RMSprop::parameters() noexcept {
  return param_groups_.at(0).params();
}

size_t RMSprop::size() const noexcept {
  // TODO: call _size_new_design once sgd PR(which contains the changes) is pushed
  size_t count = 0;
  for (const auto& group : param_groups_) {
    count += group.params().size();
  }
  return count;
}
} // namespace optim
} // namespace torch
