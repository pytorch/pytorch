#include <torch/optim/serialize.h>

#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace torch {
namespace optim {
void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const int64_t& value) {
  archive.write(key, IValue(value));
}

void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    int64_t& value) {
  IValue ivalue;
  archive.read(key, ivalue);
  value = ivalue.toInt();
}

void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const std::vector<int64_t>& steps) {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(steps.size());
  for (const auto& step : steps) {
    tensors.push_back(torch::tensor(static_cast<int64_t>(step)));
  }
  serialize(archive, key, tensors);
}

void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    std::vector<int64_t>& steps) {
  steps.clear();
  std::vector<torch::Tensor> tensors;
  serialize(archive, key, tensors);
  for (const auto& step : tensors) {
    steps.push_back(step.item<int64_t>());
  }
}

template <typename DerivedOptimizerParamState>
void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const ska::flat_hash_map<std::string, std::unique_ptr<OptimizerParamState>>& state) {
  for (const auto& item : state) {
    serialize::OutputArchive param_state_archive; // For each OptimizerParamState
    std::string tensorimpl = item.first;
    const DerivedOptimizerParamState& curr_state = static_cast<const DerivedOptimizerParamState&>(*(item.second.get()));
    curr_state.serialize(param_state_archive, c10::guts::to_string(tensorimpl));
    archive.write(key + "/" + c10::guts::to_string(tensorimpl), param_state_archive);
  }
}

template <typename DerivedOptimizerParamState>
void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    ska::flat_hash_map<std::string, std::unique_ptr<OptimizerParamState>>& state) {
  std::vector<std::string> tensorimpl_keys = archive.keys();

  for (const std::string& tensorimpl_key : tensorimpl_keys) {
    serialize::InputArchive param_state_archive;
    archive.read(key + "/" + tensorimpl_key, param_state_archive);
    DerivedOptimizerParamState param_state;
    param_state.serialize(param_state_archive);
    state[tensorimpl_key] = c10::guts::make_unique<DerivedOptimizerParamState>(param_state);
  }
}
} // namespace optim
} // namespace torch
