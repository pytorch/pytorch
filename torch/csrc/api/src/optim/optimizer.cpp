#include <torch/optim/optimizer.h>

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/ordered_dict.h>
#include <torch/types.h>

#include <string>
#include <utility>
#include <vector>

namespace torch {
namespace optim {

bool OptimizerParamGroup::has_options() const {
  return options_ != nullptr;
}

OptimizerOptions& OptimizerParamGroup::options() {
  TORCH_CHECK(has_options());
  return *options_.get();
}

const OptimizerOptions& OptimizerParamGroup::options() const {
  TORCH_CHECK(has_options());
  return *options_.get();
}

void OptimizerParamGroup::set_options(std::unique_ptr<OptimizerOptions> options) {
  options_ = std::move(options);
}

std::vector<Tensor>& OptimizerParamGroup::params() {
  return params_;
}

const std::vector<Tensor>& OptimizerParamGroup::params() const {
  return params_;
}

std::unique_ptr<OptimizerParamState> OptimizerParamState::clone() const {
  TORCH_CHECK(false,
      "clone() has not been implemented for torch::optim::OptimizerParamState. ",
      "Subclass torch::optim::OptimizerCloneableParamState<YourOptimizerParamState> ",
      "instead of torch::optim::OptimizerParamState to inherit the ability to clone.");
}

void OptimizerParamState::serialize(torch::serialize::InputArchive& archive) {
  TORCH_CHECK(false,
    "void serialize(torch::serialize::InputArchive& archive) has not been implemented for torch::optim::OptimizerParamState. ",
    "You must override it in your subclass of torch::optim::OptimizerCloneableParamState<YourOptimizerParamState>.");
}

void OptimizerParamState::serialize(torch::serialize::OutputArchive& archive) const {
  TORCH_CHECK(false,
    "void serialize(torch::serialize::OutputArchive& archive) has not been implemented for torch::optim::OptimizerParamState. ",
    "You must override it in your subclass of torch::optim::OptimizerCloneableParamState<YourOptimizerParamState>.");
}

std::unique_ptr<OptimizerOptions> OptimizerOptions::clone() const {
  TORCH_CHECK(false,
      "clone() has not been implemented for torch::optim::OptimizerOptions. ",
      "Subclass torch::optim::OptimizerCloneableOptions<YourOptimizerOptions> ",
      "instead of torch::optim::OptimizerOptions to inherit the ability to clone.");
}

void OptimizerOptions::serialize(torch::serialize::InputArchive& archive) {
  TORCH_CHECK(false,
    "void serialize(torch::serialize::InputArchive& archive) has not been implemented for torch::optim::OptimizerOptions. ",
    "You must override it in your subclass of torch::optim::OptimizerCloneableOptions<YourOptimizerOptions>.");
}

void OptimizerOptions::serialize(torch::serialize::OutputArchive& archive) const {
  TORCH_CHECK(false,
    "void serialize(torch::serialize::OutputArchive& archive) has not been implemented for torch::optim::OptimizerOptions. ",
    "You must override it in your subclass of torch::optim::OptimizerCloneableOptions<YourOptimizerOptions>.");
}

namespace detail {
OptimizerBase::OptimizerBase(std::vector<Tensor> parameters)
    : parameters_(std::move(parameters)) {}

void OptimizerBase::add_param_group(const OptimizerParamGroup& param_group) {
  for (const auto& param : param_group.params()) {
    TORCH_CHECK(param.is_leaf(), "can't optimize a non-leaf Tensor");
  }
  OptimizerParamGroup param_group_(param_group.params());
  if (!param_group.has_options()) {
    param_group_.set_options(defaults_->clone());
  } else {
    param_group_.set_options(param_group.options().clone());
  }
  for (const auto& p : param_group_.params()) {
    TORCH_CHECK(state_.count(c10::guts::to_string(p.unsafeGetTensorImpl())) == 0,
      "some parameters appear in more than one parameter group");
  }
  param_groups_.emplace_back(std::move(param_group_));
}

void OptimizerBase::add_parameters(const std::vector<Tensor>& parameters) {
  parameters_.insert(parameters_.end(), parameters.begin(), parameters.end());
}

void OptimizerBase::_add_parameters_new_design(const std::vector<Tensor>& parameters) {
  auto& parameters_ = param_groups_[0].params();
  parameters_.insert(parameters_.end(), parameters.begin(), parameters.end());
}

void OptimizerBase::zero_grad() {
  for (auto& parameter : parameters_) {
    if (parameter.grad().defined()) {
      parameter.grad().detach_();
      parameter.grad().zero_();
    }
  }
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (p.grad().defined()) {
        p.grad().detach_();
        p.grad().zero_();
      }
    }
  }
}

// TODO: remove this function after all the optimizers use the new design
const std::vector<Tensor>& OptimizerBase::parameters() const noexcept {
  return parameters_;
}

const std::vector<Tensor>& OptimizerBase::_parameters_new_design() const noexcept {
   return param_groups_.at(0).params();
}

// TODO: remove this function after all the optimizers use the new design
std::vector<Tensor>& OptimizerBase::parameters() noexcept {
  return parameters_;
}

std::vector<Tensor>& OptimizerBase::_parameters_new_design() noexcept {
   return param_groups_.at(0).params();
}

// TODO: update size to return the sum of #params in all param_groups
size_t OptimizerBase::size() const noexcept {
  return parameters_.size();
}

size_t OptimizerBase::_size_new_design() const noexcept {
  size_t count = 0;
  for (const auto& group : param_groups_) {
    count += group.params().size();
  }
  return count;
}

OptimizerOptions& OptimizerBase::defaults() noexcept {
  return *defaults_.get();
}

const OptimizerOptions& OptimizerBase::defaults() const noexcept {
  return *defaults_.get();
}

std::vector<OptimizerParamGroup>& OptimizerBase::param_groups() noexcept {
  return param_groups_;
}

const std::vector<OptimizerParamGroup>& OptimizerBase::param_groups() const noexcept {
  return param_groups_;
}

ska::flat_hash_map<std::string, std::unique_ptr<OptimizerParamState>>& OptimizerBase::state() noexcept {
  return state_;
}

const ska::flat_hash_map<std::string, std::unique_ptr<OptimizerParamState>>& OptimizerBase::state() const noexcept {
  return state_;
}

Tensor& OptimizerBase::buffer_at(std::vector<Tensor>& buffers, size_t index) {
  if (buffers.size() <= index) {
    buffers.reserve(index);
    for (auto i = buffers.size(); i <= index; ++i) {
      buffers.emplace_back(torch::zeros_like(parameters_.at(i)));
    }
  }
  // Copy the buffer to the device and dtype of the parameter.
  const auto& parameter = parameters_.at(index);
  const auto& buffer = buffers.at(index);
  if (buffer.device() != parameter.device() ||
      buffer.dtype() != parameter.dtype()) {
    buffers[index] = buffer.to(parameter.device(), parameter.scalar_type());
  }
  return buffers[index];
}

void OptimizerBase::save(serialize::OutputArchive& archive) const {}
void OptimizerBase::load(serialize::InputArchive& archive) {}

/// Serializes an `OptimizerBase` into an `OutputArchive`.
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const OptimizerBase& optimizer) {
  optimizer.save(archive);
  return archive;
}

/// Deserializes a `Tensor` from an `InputArchive`.
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    OptimizerBase& optimizer) {
  optimizer.load(archive);
  return archive;
}
} // namespace detail
} // namespace optim
} // namespace torch
