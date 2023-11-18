#include <torch/csrc/jit/mobile/train/optim/sgd.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <functional>

namespace torch {
namespace jit {
namespace mobile {

bool SGDParamGroup::has_options() const {
  return options_ != nullptr;
}

SGDOptions& SGDParamGroup::options() {
  TORCH_CHECK(has_options());
  return *options_.get();
}

const SGDOptions& SGDParamGroup::options() const {
  TORCH_CHECK(has_options());
  return *options_.get();
}

void SGDParamGroup::set_options(std::unique_ptr<SGDOptions> options) {
  options_ = std::move(options);
}

std::vector<Tensor>& SGDParamGroup::params() {
  return params_;
}

const std::vector<Tensor>& SGDParamGroup::params() const {
  return params_;
}

SGDOptions::SGDOptions(double lr) : lr_(lr) {}

bool operator==(const SGDOptions& lhs, const SGDOptions& rhs) {
  return (lhs.lr() == rhs.lr()) && (lhs.momentum() == rhs.momentum()) &&
      (lhs.dampening() == rhs.dampening()) &&
      (lhs.weight_decay() == rhs.weight_decay()) &&
      (lhs.nesterov() == rhs.nesterov());
}

bool operator==(const SGDParamState& lhs, const SGDParamState& rhs) {
  return torch::equal(lhs.momentum_buffer(), rhs.momentum_buffer());
}

void SGD::add_param_group(const SGDParamGroup& param_group) {
  for (const auto& param : param_group.params()) {
    TORCH_CHECK(param.is_leaf(), "can't optimize a non-leaf Tensor");
  }
  TORCH_INTERNAL_ASSERT(defaults_ != nullptr);
  SGDParamGroup param_group_(param_group.params());
  if (!param_group.has_options()) {
    param_group_.set_options(defaults_->clone());
  } else {
    param_group_.set_options(param_group.options().clone());
  }
  for (const auto& p : param_group_.params()) {
    TORCH_CHECK(
        state_.count(p.unsafeGetTensorImpl()) == 0,
        "some parameters appear in more than one parameter group");
  }
  param_groups_.emplace_back(std::move(param_group_));
}

void SGD::zero_grad() {
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (p.grad().defined()) {
        p.grad().detach_();
        p.grad().zero_();
      }
    }
  }
}

Tensor SGD::step(const LossClosure& closure) {
  NoGradGuard no_grad;
  Tensor loss = {};
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }
  for (auto& group : param_groups_) {
    auto& options = static_cast<SGDOptions&>(group.options());
    auto weight_decay = options.weight_decay();
    auto momentum = options.momentum();
    auto dampening = options.dampening();
    auto nesterov = options.nesterov();

    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }
      auto d_p = p.grad().data();
      if (weight_decay != 0) {
        d_p = d_p.add(p.data(), weight_decay);
      }
      if (momentum != 0) {
        Tensor buf;
        auto param_state = state_.find(p.unsafeGetTensorImpl());
        if (param_state == state_.end()) {
          buf = torch::clone(d_p).detach();
          auto state = std::make_unique<SGDParamState>();
          state->momentum_buffer(buf);
          state_[p.unsafeGetTensorImpl()] = std::move(state);
        } else {
          buf = static_cast<SGDParamState&>(*param_state->second)
                    .momentum_buffer();
          buf.mul_(momentum).add_(d_p, 1 - dampening);
        }
        if (nesterov) {
          d_p = d_p.add(buf, momentum);
        } else {
          d_p = buf;
        }
      }
      p.data().add_(d_p, -1 * options.lr());
    }
  }
  return loss;
}
} // namespace mobile
} // namespace jit
} // namespace torch
