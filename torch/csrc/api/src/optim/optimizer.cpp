#include <torch/optim/optimizer.h>

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/types.h>

// Include complete type definitions for all optimizers to enable dynamic_cast
#include <torch/optim/adagrad.h>
#include <torch/optim/adam.h>
#include <torch/optim/adamw.h>
#include <torch/optim/lbfgs.h>
#include <torch/optim/rmsprop.h>
#include <torch/optim/sgd.h>

#include <string>
#include <utility>
#include <vector>

namespace torch::optim {

// Implementation of OptimizerCloneableOptions<Derived>::_merge_by_comparison
// Moved here to anchor vtable/typeinfo for template instantiations
template <typename Derived>
void OptimizerCloneableOptions<Derived>::_merge_by_comparison(
    const Derived& defaults,
    const Derived& user_options) {
  auto* result = static_cast<Derived*>(this);
  *result = defaults; // Start with optimizer defaults

  // Create constructor defaults instance for comparison
  Derived constructor_defaults = []() {
    if constexpr (std::is_default_constructible_v<Derived>) {
      return Derived{};
    } else {
      // Handle optimizers requiring constructor parameters
      if constexpr (std::is_same_v<Derived, SGDOptions>) {
        return Derived(1e-3);
      } else if constexpr (std::is_same_v<Derived, AdagradOptions>) {
        return Derived(1e-2);
      } else if constexpr (std::is_same_v<Derived, RMSpropOptions>) {
        return Derived(1e-2);
      } else if constexpr (std::is_same_v<Derived, LBFGSOptions>) {
        return Derived(1);
      } else {
        return Derived{};
      }
    }
  }();

  // Merge fields: preserve user-set values, inherit defaults for unset values

  if constexpr (OptimizerCloneableOptions<Derived>::_has_lr<Derived>::value) {
    if (user_options.lr() != constructor_defaults.lr()) {
      result->lr(user_options.lr());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_momentum<
                    Derived>::value) {
    if (user_options.momentum() != constructor_defaults.momentum()) {
      result->momentum(user_options.momentum());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_weight_decay<
                    Derived>::value) {
    if (user_options.weight_decay() != constructor_defaults.weight_decay()) {
      result->weight_decay(user_options.weight_decay());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_dampening<
                    Derived>::value) {
    if (user_options.dampening() != constructor_defaults.dampening()) {
      result->dampening(user_options.dampening());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_nesterov<
                    Derived>::value) {
    if (user_options.nesterov() != constructor_defaults.nesterov()) {
      result->nesterov(user_options.nesterov());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_betas<
                    Derived>::value) {
    if (user_options.betas() != constructor_defaults.betas()) {
      result->betas(user_options.betas());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_eps<Derived>::value) {
    if (user_options.eps() != constructor_defaults.eps()) {
      result->eps(user_options.eps());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_amsgrad<
                    Derived>::value) {
    if (user_options.amsgrad() != constructor_defaults.amsgrad()) {
      result->amsgrad(user_options.amsgrad());
    }
  }

  // Optimizer-specific fields - automatically detected and handled
  if constexpr (OptimizerCloneableOptions<Derived>::_has_lr_decay<
                    Derived>::value) {
    if (user_options.lr_decay() != constructor_defaults.lr_decay()) {
      result->lr_decay(user_options.lr_decay());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_alpha<
                    Derived>::value) {
    if (user_options.alpha() != constructor_defaults.alpha()) {
      result->alpha(user_options.alpha());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_centered<
                    Derived>::value) {
    if (user_options.centered() != constructor_defaults.centered()) {
      result->centered(user_options.centered());
    }
  }
  if constexpr (OptimizerCloneableOptions<
                    Derived>::_has_initial_accumulator_value<Derived>::value) {
    if (user_options.initial_accumulator_value() !=
        constructor_defaults.initial_accumulator_value()) {
      result->initial_accumulator_value(
          user_options.initial_accumulator_value());
    }
  }

  // LBFGS-specific fields with appropriate types
  if constexpr (OptimizerCloneableOptions<Derived>::_has_max_iter<
                    Derived>::value) {
    if (user_options.max_iter() != constructor_defaults.max_iter()) {
      result->max_iter(user_options.max_iter());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_max_eval<
                    Derived>::value) {
    if (user_options.max_eval() != constructor_defaults.max_eval()) {
      result->max_eval(user_options.max_eval());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_tolerance_grad<
                    Derived>::value) {
    if (user_options.tolerance_grad() !=
        constructor_defaults.tolerance_grad()) {
      result->tolerance_grad(user_options.tolerance_grad());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_tolerance_change<
                    Derived>::value) {
    if (user_options.tolerance_change() !=
        constructor_defaults.tolerance_change()) {
      result->tolerance_change(user_options.tolerance_change());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_history_size<
                    Derived>::value) {
    if (user_options.history_size() != constructor_defaults.history_size()) {
      result->history_size(user_options.history_size());
    }
  }
  if constexpr (OptimizerCloneableOptions<Derived>::_has_line_search_fn<
                    Derived>::value) {
    if (user_options.line_search_fn() !=
        constructor_defaults.line_search_fn()) {
      result->line_search_fn(user_options.line_search_fn());
    }
  }
}

// Explicit template instantiations to anchor vtable/typeinfo
// These instantiations ensure the compiler generates the full class definition
// and vtable for each OptimizerCloneableOptions<T> specialization
template class OptimizerCloneableOptions<SGDOptions>;
template class OptimizerCloneableOptions<AdamOptions>;
template class OptimizerCloneableOptions<AdamWOptions>;
template class OptimizerCloneableOptions<AdagradOptions>;
template class OptimizerCloneableOptions<RMSpropOptions>;
template class OptimizerCloneableOptions<LBFGSOptions>;

// Simple implementation using variadic template helper
void Optimizer::_try_merge_all_optimizers(
    std::unique_ptr<OptimizerOptions>& final_options,
    const OptimizerOptions& user_options,
    const OptimizerOptions& defaults) {
  // Clean one-liner replaces the entire repetitive dispatch chain
  _try_merge_all_optimizer_types<
      SGDOptions,
      AdamOptions,
      AdamWOptions,
      AdagradOptions,
      RMSpropOptions,
      LBFGSOptions>(final_options, user_options, defaults);
}

bool OptimizerParamGroup::has_options() const {
  return options_ != nullptr;
}

OptimizerOptions& OptimizerParamGroup::options() {
  TORCH_CHECK(has_options());
  return *options_;
}

const OptimizerOptions& OptimizerParamGroup::options() const {
  TORCH_CHECK(has_options());
  return *options_;
}

void OptimizerParamGroup::set_options(
    std::unique_ptr<OptimizerOptions> options) {
  options_ = std::move(options);
}

std::vector<Tensor>& OptimizerParamGroup::params() {
  return params_;
}

const std::vector<Tensor>& OptimizerParamGroup::params() const {
  return params_;
}

std::unique_ptr<OptimizerParamState> OptimizerParamState::clone() const {
  TORCH_CHECK(
      false,
      "clone() has not been implemented for torch::optim::OptimizerParamState. ",
      "Subclass torch::optim::OptimizerCloneableParamState<YourOptimizerParamState> ",
      "instead of torch::optim::OptimizerParamState to inherit the ability to clone.");
}

void OptimizerParamState::serialize(torch::serialize::InputArchive& archive) {
  TORCH_CHECK(
      false,
      "void serialize(torch::serialize::InputArchive& archive) has not been implemented for torch::optim::OptimizerParamState. ",
      "You must override it in your subclass of torch::optim::OptimizerCloneableParamState<YourOptimizerParamState>.");
}

void OptimizerParamState::serialize(
    torch::serialize::OutputArchive& archive) const {
  TORCH_CHECK(
      false,
      "void serialize(torch::serialize::OutputArchive& archive) has not been implemented for torch::optim::OptimizerParamState. ",
      "You must override it in your subclass of torch::optim::OptimizerCloneableParamState<YourOptimizerParamState>.");
}

double OptimizerOptions::get_lr() const {
  TORCH_CHECK(
      false,
      "double get_lr() has not been overridden and implemented in subclass of torch::optim::OptimizerOptions, you must override it in your subclass.");
}

void OptimizerOptions::set_lr(const double lr) {
  TORCH_CHECK(
      false,
      "double set_lr() has not been overridden and implemented in subclass of torch::optim::OptimizerOptions, you must override it in your subclass.");
}

std::unique_ptr<OptimizerOptions> OptimizerOptions::clone() const {
  TORCH_CHECK(
      false,
      "clone() has not been implemented for torch::optim::OptimizerOptions. ",
      "Subclass torch::optim::OptimizerCloneableOptions<YourOptimizerOptions> ",
      "instead of torch::optim::OptimizerOptions to inherit the ability to clone.");
}

void OptimizerOptions::serialize(torch::serialize::InputArchive& archive) {
  TORCH_CHECK(
      false,
      "void serialize(torch::serialize::InputArchive& archive) has not been implemented for torch::optim::OptimizerOptions. ",
      "You must override it in your subclass of torch::optim::OptimizerCloneableOptions<YourOptimizerOptions>.");
}

void OptimizerOptions::serialize(
    torch::serialize::OutputArchive& archive) const {
  TORCH_CHECK(
      false,
      "void serialize(torch::serialize::OutputArchive& archive) has not been implemented for torch::optim::OptimizerOptions. ",
      "You must override it in your subclass of torch::optim::OptimizerCloneableOptions<YourOptimizerOptions>.");
}

void Optimizer::add_param_group(const OptimizerParamGroup& param_group) {
  for (const auto& param : param_group.params()) {
    TORCH_CHECK(param.is_leaf(), "can't optimize a non-leaf Tensor");
  }
  TORCH_INTERNAL_ASSERT(defaults_ != nullptr);
  OptimizerParamGroup param_group_(param_group.params());
  if (!param_group.has_options()) {
    // No options provided - use defaults directly
    param_group_.set_options(defaults_->clone());
  } else {
    // Options provided - merge user's explicit settings with defaults for
    // parameter group inheritance This enables Python-C++ API parity by
    // honoring user intent while inheriting missing parameters
    auto final_options = defaults_->clone();

    // Simple variadic dispatch - try all known optimizer types
    _try_merge_all_optimizers(final_options, param_group.options(), *defaults_);

    // If no merging was done (custom optimizer), final_options already contains
    // defaults
    param_group_.set_options(std::move(final_options));
  }
  for (const auto& p : param_group_.params()) {
    TORCH_CHECK(
        state_.count(p.unsafeGetTensorImpl()) == 0,
        "some parameters appear in more than one parameter group");
  }
  param_groups_.emplace_back(std::move(param_group_));
}

void Optimizer::add_parameters(const std::vector<Tensor>& parameters) {
  TORCH_WARN("Optimizer::add_parameters() will be removed in PyTorch 1.6");
  auto& parameters_ = param_groups_[0].params();
  parameters_.insert(parameters_.end(), parameters.begin(), parameters.end());
}

void Optimizer::zero_grad(bool set_to_none) {
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (p.mutable_grad().defined()) {
        p.mutable_grad().detach_();
        if (set_to_none)
          p.mutable_grad().reset();
        else
          p.mutable_grad().zero_();
      }
    }
  }
}

const std::vector<Tensor>& Optimizer::parameters() const noexcept {
  TORCH_WARN("Optimizer::parameters() will be removed in PyTorch 1.6");
  return param_groups_.at(0).params();
}

std::vector<Tensor>& Optimizer::parameters() noexcept {
  TORCH_WARN("Optimizer::parameters() will be removed in PyTorch 1.6");
  return param_groups_.at(0).params();
}

size_t Optimizer::size() const noexcept {
  TORCH_WARN("Optimizer::size() will be removed in PyTorch 1.6");
  size_t count = 0;
  for (const auto& group : param_groups_) {
    count += group.params().size();
  }
  return count;
}

OptimizerOptions& Optimizer::defaults() noexcept {
  return *defaults_;
}

const OptimizerOptions& Optimizer::defaults() const noexcept {
  return *defaults_;
}

std::vector<OptimizerParamGroup>& Optimizer::param_groups() noexcept {
  return param_groups_;
}

const std::vector<OptimizerParamGroup>& Optimizer::param_groups()
    const noexcept {
  return param_groups_;
}

ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>& Optimizer::
    state() noexcept {
  return state_;
}

const ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>&
Optimizer::state() const noexcept {
  return state_;
}

void Optimizer::save(serialize::OutputArchive& archive) const {}
void Optimizer::load(serialize::InputArchive& archive) {}

/// Serializes an `Optimizer` into an `OutputArchive`.
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Optimizer& optimizer) {
  optimizer.save(archive);
  return archive;
}

/// Deserializes a `Tensor` from an `InputArchive`.
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Optimizer& optimizer) {
  optimizer.load(archive);
  return archive;
}

} // namespace torch::optim
