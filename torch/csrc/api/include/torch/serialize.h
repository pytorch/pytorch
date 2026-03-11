#pragma once

#include <c10/util/irange.h>
#include <torch/csrc/Export.h>
#include <torch/serialize/archive.h>
#include <torch/serialize/tensor.h>

#include <utility>

namespace torch {

/// Serializes the given `value`.
template <typename Value, typename... SaveToArgs>
void save(const Value& value, SaveToArgs&&... args) {
  serialize::OutputArchive archive(std::make_shared<jit::CompilationUnit>());
  archive << value;
  archive.save_to(std::forward<SaveToArgs>(args)...);
}

/// Serializes the given `tensor_vec` of type `std::vector<torch::Tensor>`.
template <typename... SaveToArgs>
void save(const std::vector<torch::Tensor>& tensor_vec, SaveToArgs&&... args) {
  serialize::OutputArchive archive(std::make_shared<jit::CompilationUnit>());
  for (const auto i : c10::irange(tensor_vec.size())) {
    auto& value = tensor_vec[i];
    archive.write(std::to_string(i), value);
  }
  archive.save_to(std::forward<SaveToArgs>(args)...);
}

/// Saves the state dictionary (parameters and buffers) of a module.
template <typename Module, typename... SaveToArgs>
void save_state_dict(const Module& module, SaveToArgs&&... args) {
  serialize::OutputArchive archive(std::make_shared<jit::CompilationUnit>());
  
  // Save parameters
  for (const auto& param : module->named_parameters()) {
    archive.write(param.key(), param.value());
  }
  
  // Save buffers
  for (const auto& buffer : module->named_buffers()) {
    archive.write(buffer.key(), buffer.value());
  }
  
  archive.save_to(std::forward<SaveToArgs>(args)...);
}

TORCH_API std::vector<char> pickle_save(const torch::IValue& ivalue);
TORCH_API torch::IValue pickle_load(const std::vector<char>& data);

/// Deserializes the given `value`.
template <typename Value, typename... LoadFromArgs>
void load(Value& value, LoadFromArgs&&... args) {
  serialize::InputArchive archive;
  archive.load_from(std::forward<LoadFromArgs>(args)...);
  archive >> value;
}

/// Deserializes the given `tensor_vec` of type `std::vector<torch::Tensor>`.
template <typename... LoadFromArgs>
void load(std::vector<torch::Tensor>& tensor_vec, LoadFromArgs&&... args) {
  serialize::InputArchive archive;
  archive.load_from(std::forward<LoadFromArgs>(args)...);

  size_t index = 0;
  torch::Tensor value;
  while (archive.try_read(std::to_string(index), value)) {
    tensor_vec.push_back(std::move(value));
    value = torch::Tensor();
    index++;
  }
}

/// Loads the state dictionary (parameters and buffers) into a module.
template <typename Module, typename... LoadFromArgs>
void load_state_dict(Module& module, LoadFromArgs&&... args) {
  serialize::InputArchive archive;
  archive.load_from(std::forward<LoadFromArgs>(args)...);
  
  // Load parameters
  for (auto& param : module->named_parameters()) {
    torch::Tensor loaded_param;
    if (archive.try_read(param.key(), loaded_param)) {
      param.value().copy_(loaded_param);
    }
  }
  
  // Load buffers
  for (auto& buffer : module->named_buffers()) {
    torch::Tensor loaded_buffer;
    if (archive.try_read(buffer.key(), loaded_buffer)) {
      buffer.value().copy_(loaded_buffer);
    }
  }
}

} // namespace torch
