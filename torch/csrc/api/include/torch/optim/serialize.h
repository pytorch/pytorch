#pragma once

#include <torch/serialize/archive.h>
#include <torch/types.h>
#include <torch/optim/optimizer.h>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace torch {
namespace optim {
namespace detail {
  // Utility function to save state
  template <typename DerivedOptimizerParamState>
  void serialize(
      serialize::OutputArchive& archive,
      const ska::flat_hash_map<std::string, std::unique_ptr<OptimizerParamState>>& state) {
    for (const auto& item : state) {
      serialize::OutputArchive param_state_archive(archive.compilation_unit());
      std::string tensorimpl_key = item.first;
      const DerivedOptimizerParamState& curr_state = static_cast<const DerivedOptimizerParamState&>(*(item.second.get()));
      curr_state.serialize(param_state_archive);
      archive.write(tensorimpl_key, param_state_archive);
    }
  }

  // Utility function to load state
  template <typename DerivedOptimizerParamState>
  void serialize(
      serialize::InputArchive& archive,
      ska::flat_hash_map<std::string, std::unique_ptr<OptimizerParamState>>& state) {
    std::vector<std::string> tensorimpl_keys = archive.keys();
    for (const std::string& tensorimpl_key : tensorimpl_keys) {
      serialize::InputArchive param_state_archive;
      archive.read(tensorimpl_key, param_state_archive);
      DerivedOptimizerParamState param_state;
      param_state.serialize(param_state_archive);
      state[tensorimpl_key] = std::make_unique<DerivedOptimizerParamState>(param_state);
    }
  }

  // Utility function to save param_groups
  template <typename DerivedOptimizerParamOptions>
  void serialize(
      serialize::OutputArchive& archive,
      const std::vector<OptimizerParamGroup>& param_groups) {
    archive.write("param_groups/size", torch::tensor(static_cast<int64_t>(param_groups.size())));
    for (size_t i = 0; i < param_groups.size(); i++) {
      serialize::OutputArchive param_group_archive(archive.compilation_unit());
      std::vector<Tensor> params = param_groups[i].params();
      param_group_archive.write(
          "params/size", torch::tensor(static_cast<int64_t>(params.size())));
      for (size_t index = 0; index < params.size(); index++) {
        param_group_archive.write(
            "params/" + c10::guts::to_string(index), IValue(c10::guts::to_string(params[index].unsafeGetTensorImpl())));
      }
      const DerivedOptimizerParamOptions& param_group_options = static_cast<const DerivedOptimizerParamOptions&>(param_groups[i].options());
      serialize::OutputArchive param_group_options_archive(param_group_archive.compilation_unit());
      param_group_options.serialize(param_group_options_archive);
      param_group_archive.write("options", param_group_options_archive);
      archive.write("param_groups/" + c10::guts::to_string(i), param_group_archive);
    }
  }

  // Utility function to load param_groups
  // We take as input vector of pair of string and unique_ptr to optimizer options so that we can retain the state
  // for each param by using the old tensor impl keys (saved during serialization) and map the new tensor impl keys to
  // the correct state for each param
  template <typename DerivedOptimizerParamOptions>
  void serialize(
      serialize::InputArchive& archive,
      std::vector<std::pair<std::vector<std::string>, std::unique_ptr<OptimizerOptions>>>& param_groups) {
    torch::Tensor param_groups_size_tensor;
    archive.read("param_groups/size", param_groups_size_tensor);
    const int64_t param_groups_size = param_groups_size_tensor.item<int64_t>();
    for (int64_t i = 0; i < param_groups_size; i++) {
      serialize::InputArchive param_group_archive;
      archive.read("param_groups/" + c10::guts::to_string(i), param_group_archive);
      torch::Tensor size_tensor;
      param_group_archive.read("params/size", size_tensor);
      const int64_t size = size_tensor.item<int64_t>();
      std::vector<std::string> params;
      for (int64_t index = 0; index < size; ++index) {
        IValue ivalue;
        param_group_archive.read(
          "params/" + c10::to_string(index), ivalue);
        std::string element = ivalue.toStringRef();
        params.emplace_back(element);
      }
      serialize::InputArchive param_group_options_archive;
      param_group_archive.read("options", param_group_options_archive);
      DerivedOptimizerParamOptions param_group_options(0);
      param_group_options.serialize(param_group_options_archive);
      param_groups.emplace_back(std::make_pair(params, std::make_unique<DerivedOptimizerParamOptions>(param_group_options)));
    }
  }
} // namespace detail


// Note: These functions are all called `serialize()` so they can be called
// inside a template where the archive type is a template type and can thus be
// passed such that the appropriate overload is selected.

/// Utility function to save a value of `int64_t` type.
void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const int64_t& value);

/// Utility function to load a value of `int64_t` type.
void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    int64_t& value);

/// Utility function to save a vector of step buffers.
void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const std::vector<int64_t>& steps);

/// Utility function to load a vector of step buffers.
void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    std::vector<int64_t>& steps);

// Utility function to save state and param_groups
template <typename DerivedOptimizerParamState, typename DerivedOptimizerParamOptions>
void serialize(
    serialize::OutputArchive& archive,
    const detail::OptimizerBase& optimizer) {
  archive.write("pytorch_version", IValue("1.5.0"));
  serialize::OutputArchive state_archive(archive.compilation_unit());
  detail::serialize<DerivedOptimizerParamState>(state_archive, optimizer.state());
  archive.write("state", state_archive);

  serialize::OutputArchive param_groups_archive(archive.compilation_unit());
  detail::serialize<DerivedOptimizerParamOptions>(param_groups_archive, optimizer.param_groups());
  archive.write("param_groups", param_groups_archive);
}

// Utility function to load state and param_groups and update state
template <typename DerivedOptimizerParamState, typename DerivedOptimizerParamOptions>
void serialize(
    serialize::InputArchive& archive,
    detail::OptimizerBase& optimizer) {

    IValue pytorch_version;
    archive.read("pytorch_version", pytorch_version);
    TORCH_INTERNAL_ASSERT(pytorch_version.toStringRef() == "1.5.0");
    serialize::InputArchive state_archive;
    archive.read("state", state_archive);
    ska::flat_hash_map<std::string, std::unique_ptr<OptimizerParamState>> saved_state;
    detail::serialize<DerivedOptimizerParamState>(state_archive, saved_state);

    serialize::InputArchive param_groups_archive;
    archive.read("param_groups", param_groups_archive);
    std::vector<std::pair<std::vector<std::string>, std::unique_ptr<OptimizerOptions>>> saved_param_groups;
    detail::serialize<DerivedOptimizerParamOptions>(param_groups_archive, saved_param_groups);

    // update state
    TORCH_CHECK(saved_param_groups.size() == optimizer.param_groups().size(), "loaded state dict has a different number of parameter groups");
    for (size_t i = 0; i < saved_param_groups.size(); i++) {
      std::vector<std::string> param_group_old_keys = saved_param_groups[i].first;
      std::vector<Tensor> params = optimizer.param_groups()[i].params();
      TORCH_CHECK(param_group_old_keys.size() == params.size(), "loaded state dict contains a parameter group that has a different size than the optimizer's parameter group");

      for (size_t idx = 0; idx < params.size(); idx++) {
        if(saved_state.find(param_group_old_keys[idx]) != saved_state.end()) {
          optimizer.state()[c10::guts::to_string(params[idx].unsafeGetTensorImpl())] = std::move(saved_state[param_group_old_keys[idx]]);
        }
      }
    }
}

/// Utility function to save a vector of buffers.
template <typename BufferContainer>
void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const BufferContainer& buffers) {
  archive.write(
      key + "/size", torch::tensor(static_cast<int64_t>(buffers.size())));
  for (size_t index = 0; index < buffers.size(); ++index) {
    archive.write(
        key + "/" + c10::to_string(index), buffers[index], /*is_buffer=*/true);
  }
}

/// Utility function to load a vector of buffers.
template <typename BufferContainer>
void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    BufferContainer& buffers) {
  buffers.clear();
  torch::Tensor size_tensor;
  archive.read(key + "/size", size_tensor);
  const size_t size = size_tensor.item<int64_t>();
  for (size_t index = 0; index < size; ++index) {
    buffers.emplace_back();
    archive.read(
        key + "/" + c10::to_string(index), buffers.back(), /*is_buffer=*/true);
  }
}

template <typename T>
c10::List<T> deque_to_list(const std::deque<T>& dq) {
  c10::List<T> list;
  list.reserve(dq.size());
  for (const auto& e : dq) {
    list.emplace_back(e);
  }
  return list;
}

template <typename T>
std::deque<T> list_to_deque(const c10::List<T>& list) {
  std::deque<T> dq;
  for (const auto& e : list) {
    dq.emplace_back(e);
  }
  return dq;
}

#define _TORCH_OPTIM_SERIALIZE(name) \
  torch::optim::serialize(archive, #name, self.name)

#define _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(OptimizerName) \
  torch::optim::serialize<OptimizerName##ParamState, OptimizerName##Options>(archive, self)

#define _TORCH_OPTIM_SERIALIZE_TORCH_ARG(name) { \
  auto ivalue = torch::IValue(name()); \
  /* do not serialize if name is an undefined tensor*/ \
  if (!(ivalue.isTensor() && ivalue.unsafeToTensorImpl() == at::UndefinedTensorImpl::singleton())) { \
    archive.write(#name, ivalue); \
  } \
}

#define _TORCH_OPTIM_SERIALIZE_TORCH_ARG_DEQUE(name) { \
  c10::IValue ivalue = torch::IValue(deque_to_list(name())); \
  archive.write(#name, ivalue); \
}

#define _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(T, name) { \
  c10::IValue ivalue; \
  bool exists = archive.try_read(#name, ivalue); \
  if (exists) {\
    name(ivalue.to<T>()); \
  } else { \
    bool is_tensor_type = std::is_base_of<torch::Tensor, T>::value; \
    TORCH_INTERNAL_ASSERT(is_tensor_type); \
  } \
}

#define _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_OPTIONAL(T, name) { \
  c10::IValue ivalue; \
  bool exists = archive.try_read(#name, ivalue); \
  if (exists) { \
    name(ivalue.toOptional<T>()); \
  } \
}

#define _TORCH_OPTIM_DESERIALIZE_TORCH_ARG_DEQUE(T, name) { \
  c10::IValue ivalue; \
  archive.read(#name, ivalue); \
  auto list = ivalue.to<c10::List<T::value_type>>(); \
  name(list_to_deque(list)); \
}

} // namespace optim
} // namespace torch
