#pragma once

#include <c10/util/irange.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/serialize/archive.h>
#include <torch/serialize/tensor.h>

#include <utility>
#include <ATen/core/stack.h>

namespace torch {

/// Serializes the given `value`.
/// There must be an overload of `operator<<` between `serialize::OutputArchive`
/// and `Value` for this method to be well-formed. Currently, such an overload
/// is provided for (subclasses of):
///
/// - `torch::nn::Module`,
/// - `torch::optim::Optimizer`
/// - `torch::Tensor`
///
/// To perform the serialization, a `serialize::OutputArchive` is constructed,
/// and all arguments after the `value` are forwarded to its `save_to` method.
/// For example, you can pass a filename, or an `ostream`.
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::Linear model(3, 4);
///   torch::save(model, "model.pt");
///
///   torch::optim::SGD sgd(/*lr=*/0.9);
///   std::ostringstream stream;
///   // Note that the same stream cannot be used in multiple torch::save(...)
///   // invocations, otherwise the header will be corrupted.
///   torch::save(sgd, stream);
///
///   auto tensor = torch::ones({3, 4});
///   torch::save(tensor, "my_tensor.pt");
/// \endrst
// template <typename Value = std::enable_if<false, long>, typename...
// SaveToArgs>
template <
    typename Value,
    typename... SaveToArgs,
    typename std::enable_if<
        !std::is_same<Value, torch::IValue>::value &&
            !std::is_same<Value, std::vector<torch::Tensor>&>::value &&
            !std::is_same<Value, const std::vector<torch::Tensor>&>::value &&
            !std::is_same<Value, std::vector<torch::Tensor>>::value,
        Value>::type* = nullptr>
void save(const Value& value, SaveToArgs&&... args) {
  serialize::OutputArchive archive(std::make_shared<jit::CompilationUnit>());
  archive << value;
  archive.save_to(std::forward<SaveToArgs>(args)...);
}

/// Serializes the given `tensor_vec` of type `std::vector<torch::Tensor>`.
///
/// To perform the serialization, a `serialize::OutputArchive` is constructed,
/// and all arguments after the `tensor_vec` are forwarded to its `save_to`
/// method. For example, you can pass a filename, or an `ostream`.
///
/// \rst
/// .. code-block:: cpp
///
///   std::vector<torch::Tensor> tensor_vec = { torch::randn({1, 2}),
///   torch::randn({3, 4}) }; torch::save(tensor_vec, "my_tensor_vec.pt");
///
///   std::vector<torch::Tensor> tensor_vec = { torch::randn({5, 6}),
///   torch::randn({7, 8}) }; std::ostringstream stream;
///   // Note that the same stream cannot be used in multiple torch::save(...)
///   // invocations, otherwise the header will be corrupted.
///   torch::save(tensor_vec, stream);
/// \endrst
template <typename... SaveToArgs>
void save(const std::vector<torch::Tensor>& tensor_vec, SaveToArgs&&... args) {
  serialize::OutputArchive archive(std::make_shared<jit::CompilationUnit>());
  for (const auto i : c10::irange(tensor_vec.size())) {
    auto& value = tensor_vec[i];
    archive.write(c10::to_string(i), value);
  }
  archive.save_to(std::forward<SaveToArgs>(args)...);
}

/// Deserializes the given `value`.
/// There must be an overload of `operator>>` between `serialize::InputArchive`
/// and `Value` for this method to be well-formed. Currently, such an overload
/// is provided for (subclasses of):
///
/// - `torch::nn::Module`,
/// - `torch::optim::Optimizer`
/// - `torch::Tensor`
///
/// To perform the serialization, a `serialize::InputArchive` is constructed,
/// and all arguments after the `value` are forwarded to its `load_from` method.
/// For example, you can pass a filename, or an `istream`.
///
/// \rst
/// .. code-block:: cpp
///
///   torch::nn::Linear model(3, 4);
///   torch::load(model, "model.pt");
///
///   torch::optim::SGD sgd(/*lr=*/0.9);
///   std::istringstream stream("...");
///   torch::load(sgd, stream);
///
///   auto tensor = torch::ones({3, 4});
///   torch::load(tensor, "my_tensor.pt");
/// \endrst
template <
    typename Value,
    typename... LoadFromArgs,
    typename std::enable_if<
        !std::is_same<Value, std::string>::value &&
            !std::is_same<Value, const char*>::value &&
            !std::is_same<Value, std::vector<torch::Tensor>&>::value &&
            !std::is_same<Value, const std::vector<torch::Tensor>&>::value &&
            !std::is_same<Value, std::vector<torch::Tensor>>::value,
        Value>::type* = nullptr>
void load(Value& value, LoadFromArgs&&... args) {
  serialize::InputArchive archive;
  archive.load_from(std::forward<LoadFromArgs>(args)...);
  archive >> value;
}

/// Deserializes the given `tensor_vec` of type `std::vector<torch::Tensor>`.
///
/// To perform the serialization, a `serialize::InputArchive` is constructed,
/// and all arguments after the `value` are forwarded to its `load_from` method.
/// For example, you can pass a filename, or an `istream`.
///
/// \rst
/// .. code-block:: cpp
///
///   std::vector<torch::Tensor> tensor_vec;
///   torch::load(tensor_vec, "my_tensor_vec.pt");
///
///   std::vector<torch::Tensor> tensor_vec;
///   std::istringstream stream("...");
///   torch::load(tensor_vec, stream);
/// \endrst
template <typename... LoadFromArgs>
void load(std::vector<torch::Tensor>& tensor_vec, LoadFromArgs&&... args) {
  serialize::InputArchive archive;
  archive.load_from(std::forward<LoadFromArgs>(args)...);

  // NOTE: The number of elements in the serialized `std::vector<torch::Tensor>`
  // is not known ahead of time, so we need a while-loop to increment the index,
  // and use `archive.try_read(...)` to check whether we have reached the end of
  // the serialized `std::vector<torch::Tensor>`.
  size_t index = 0;
  torch::Tensor value;
  while (archive.try_read(c10::to_string(index), value)) {
    tensor_vec.push_back(std::move(value));
    value = torch::Tensor();
    index++;
  }
}

// These are identical to the save() and load() with the same signature, but are
// kept around here for backwards compat
TORCH_API std::vector<char> pickle_save(const torch::IValue& ivalue);
TORCH_API torch::IValue pickle_load(const std::vector<char>& data);

TORCH_API std::vector<char> save(const torch::IValue& ivalue);
TORCH_API torch::IValue load(const std::vector<char>& data);

/// Serializes the given `ivalue` to `filename`.
///
/// The data saved to `filename` is compatible with `torch.save` in Python, and
/// can be loaded with `torch.load(filename)`. For backwards compatibility
/// reasons, the type passed for `ivalue` MUST be of type `torch::IValue` (i.e.
/// no conversions required).
///
/// \rst
/// .. code-block:: cpp
///
///   auto input = torch::ones({2, 2})
///   torch::save(torch::IValue(input), "my_file.pt")
///
/// \endrst
///
/// Load in Python:
///
/// \rst
/// .. code-block::
///
///   tensor = torch.load("my_file.pt")
///
/// \endrst
///
/// Load in C++:
///
/// \rst
/// .. code-block:: cpp
///
///   auto tensor = torch::load(std::string("my_file.pt")).toTensor();
///
/// \endrst

TORCH_API void save(const torch::IValue& ivalue, const std::string& filename);

/// See torch::save
TORCH_API torch::IValue load(const std::string& filename);

} // namespace torch
