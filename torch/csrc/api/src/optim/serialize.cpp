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

  serialize::OutputArchive state_archive;
  for (const auto& item : state) {
    serialize::OutputArchive param_state_archive; // For each OptimizerParamState
    std::string tensorimpl = item.first;
    const DerivedOptimizerParamState& curr_state = static_cast<const DerivedOptimizerParamState&>(*(item.second.get()));
    curr_state.serialize(param_state_archive);
    state_archive.write(tensorimpl, param_state_archive);
  }
  archive.write(key, state_archive);
}

template <typename DerivedOptimizerParamState>
void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    ska::flat_hash_map<std::string, std::unique_ptr<OptimizerParamState>>& state) {

  serialize::InputArchive state_archive;
  archive.read(key, state_archive);
  std::vector<std::string> tensorimpl_keys = state_archive.keys();
  for (const std::string& tensorimpl_key : tensorimpl_keys) {
    serialize::InputArchive param_state_archive;
    archive.read(tensorimpl_key, param_state_archive);
    DerivedOptimizerParamState param_state;
    param_state.serialize(param_state_archive);
    state[tensorimpl_key] = c10::guts::make_unique<DerivedOptimizerParamState>(param_state);
  }
}

void serialize(
  serialize::InputArchive& archive,
  const std::string& key,
  std::vector<OptimizerParamGroup> param_groups_) {
  std::vector<std::pair<std::vector<std::string>, OptimizerOptions>> param_groups;
  // for(size_t i=0; i<param_groups.size(); i++) {
  //   param_groups.push_back()
  // }
  serialize(archive, key, param_groups);
}

template <typename DerivedOptimizerParamOptions>
void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const std::vector<OptimizerParamGroup>& param_groups) {

  serialize::OutputArchive param_groups_archive;
  for (size_t i = 0; i < param_groups.size(); i++) {
    serialize::OutputArchive param_group_archive; // For each OptimizerParamState
    std::vector<Tensor> params = param_groups[i].params();
    param_group_archive.write(
        "params/size", torch::tensor(static_cast<int64_t>(params.size())));
    for(size_t index = 0; index < params.size(); index++) {
      param_group_archive.write(
          "params/" + c10::guts::to_string(index), IValue(c10::guts::to_string(params[index].unsafeGetTensorImpl())));
    }

    const DerivedOptimizerParamOptions& param_group_options = static_cast<const DerivedOptimizerParamOptions&>(param_groups[i].options());
    param_group_options.serialize(param_group_archive);
    param_group_archive.write("options", param_group_archive);
    param_groups_archive.write(key + "/" + c10::guts::to_string(i), param_group_archive);
  }
  //serialize param_groups "params"->vector<string>, "options" -> options
  archive.write(key, param_groups_archive);
}

template <typename DerivedOptimizerParamOptions>
void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    std::vector<std::pair<std::vector<std::string>, OptimizerOptions>>& param_groups) {

  serialize::InputArchive param_groups_archive;
  archive.read(key, param_groups_archive);
  //"i"->param group archive -> "params", ("options"->"learning_rate", ...)
  std::vector<std::string> indices = param_groups_archive.keys();
  for (const std::string& param_group_key : indices) {
    serialize::InputArchive param_group_archive;
    param_groups_archive.read(param_group_key, param_group_archive); //confirm

    torch::Tensor size_tensor;
    param_group_archive.read("params/size", size_tensor);
    const size_t size = size_tensor.item<int64_t>();
    std::vector<std::string> params;
    for (size_t index = 0; index < size; ++index) {
      IValue ivalue;
      param_group_archive.read(
        "params/" + c10::to_string(index), ivalue);
      std::string element = ivalue.toStringRef();
      params.push_back(element);
    }
    DerivedOptimizerParamOptions param_group_options;
    param_group_options.serialize(param_groups_archive);

    param_groups.push_back(std::make_pair(params, param_group_options));
  }
}
} // namespace optim
} // namespace torch
