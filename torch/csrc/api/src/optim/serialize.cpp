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

// void serialize(
//     serialize::OutputArchive& archive,
//     const std::string& key,
//     const ska::flat_hash_map<at::TensorImpl*, std::unique_ptr<OptimizerParamState>>& state) {
//   for (const auto& item : state) {
//     serialize::OutputArchive param_state_archive; // For each OptimizerParamState
//     at::TensorImpl* tensorimpl_ptr = item.first;
//     const OptimizerParamState& curr_state = *(item.second.get());
//     //#define stringitem(item) #item
//     curr_state.serialize(param_state_archive, c10::to_string(tensorimpl_ptr));
//     archive.write((key + "/" + c10::to_string(tensorimpl_ptr), param_state_archive);
//     //#undef stringitem
//   }
// }

// template <typename DerivedOptimizerParamState>
// void serialize(
//     serialize::InputArchive& archive,
//     const std::string& key,
//     ska::flat_hash_map<at::TensorImpl*, std::unique_ptr<OptimizerParamState>>& state) {
//   std::vector<std::string> tensorimpl_keys = archive.keys();
//   for (const std::string& tensorimpl_key : tensorimpl_keys) {
//     serialize::InputArchive param_state_archive;
//     archive.read(key + "/" + tensorimpl_key, param_state_archive);
//     OptimizerParamState param_state;
//     // NOTE: the expectation is that `DerivedOptimizerParamState::serialize` deserializes the param state `param_state`
//     // from `param_state_archive`
//     param_state.serialize(param_state_archive);
//     // NOTE: `(at::TensorImpl*)key` might not be the right way to convert `std::string` to an `at::TensorImpl*`,
//     // and we might actually want to store a string representation of the `at::TensorImpl*` pointer instead of the
//     // pointer itself in ska::flat_hash_map instead. i.e. Changing `state_`'s type in OptimizerBase from
//     // `ska::flat_hash_map<at::TensorImpl*, std::unique_ptr<OptimizerParamState>>`
//     // to
//     // `ska::flat_hash_map<std::string, std::unique_ptr<OptimizerParamState>>`
//     // We can use this https://stackoverflow.com/a/785016to convert a pointer to its string representation.
//     state[(at::TensorImpl*)tensorimpl_key] = c10::guts::make_unique<DerivedOptimizerParamState>(param_state);
//   }
// }
} // namespace optim
} // namespace torch
