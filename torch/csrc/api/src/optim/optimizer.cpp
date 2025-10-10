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
