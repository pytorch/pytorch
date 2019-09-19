#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/pickler.h>


namespace torch {
namespace jit {

/// Pickle an IValue by calling a function to handle writing the data.
///
/// `writer` is a function that takes in a pointer to a chunk of memory and its
/// size and consumes it.
///
/// See `jit::pickle` for more details.
TORCH_API void pickle(
    std::function<void(const char* data_start, size_t data_len)> writer,
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table = nullptr);

/// Save a `torch::IValue` in a format compatible with Python's `pickle` module
///
/// If present, `tensor_table` is a pointer to a table in which tensors that
/// are contained within `ivalue` are stored, and the bytes returned by the
/// pickler will only include references to these tensors in the table. This can
/// be used to keep the binary blob size small.
TORCH_API std::vector<char> pickle(
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table = nullptr);

/// Save a `torch::IValue` in a format compatible with Python's `pickle` module
///
/// Pickled values can be loaded in Python and C++:
///
/// \rst
/// .. code-block:: cpp
///
///   // Generate a tensor and save it
///   auto x = torch::ones({2, 3});
///   auto data = torch::pickle_save(x);
///   std::ofstream out("my_tensor.pt");
///   out.write(data.data(), data.size());
///   out.flush();
///
///   // Read the tensor back in in C++
///   std::ifstream input("my_tensor.pt");
///   torch::IValue ivalue = torch::pickle_load([&](char* buffer, size_t len) {
///     if (!input.good()) {
///       return false;
///     }
///     input.read(buffer, len);
///     return input.good();
///   });
///   std::cout << ivalue << "\n";
///
/// .. code-block:: python
///
///   // Read the tensor in Python
///   values = torch.load('my_tensor.pt')
///   print(values)
///
/// \endrst
TORCH_API std::vector<char> pickle_save(const IValue& ivalue);
  std::ifstream input("my_tensor.pt");
  torch::IValue ivalue = torch::pickle_load([&](char* buffer, size_t len) {
    if (!input.good()) {
      return false;
    }
    input.read(buffer, len);
    return input.good();
  });

/// Load a `torch::IValue` from a pickle file. The pickle file can be one
/// produced by `torch.save()` (for the values supported by IValue) or
/// `pickle_save`.
///
/// .. code-block:: python
///
///   // Create a tensor in Python
///   x = torch.ones(3, 3) + 10
///   torch.save(x, "my_tensor.pt")
///
/// \rst
/// .. code-block:: cpp
///
///   // Load the tensor in C++
///   std::ifstream input("my_tensor.pt");
///   torch::IValue ivalue = torch::pickle_load([&](char* buffer, size_t len) {
///     if (!input.good()) {
///       return false;
///     }
///     input.read(buffer, len);
///     return input.good();
///   });
///   std::cout << ivalue << "\n";
///
/// \endrst
TORCH_API IValue pickle_load(const std::function<bool(char*, size_t)>& reader);


/// `reader` is a function that takes in a size to read from some pickled
/// binary. `reader` should remember where it last read, and return
/// false if the read was not successful.
/// See `torch::pickle` for details.
TORCH_API IValue unpickle(
    std::function<bool(char*, size_t)> reader,
    ClassResolver class_resolver,
    const std::vector<at::Tensor>* tensor_table);

/// Decode a chunk of memory containing pickled data into its `torch::IValue`s.
///
/// If any `torch::IValue`s in the pickled data are `Object`s, then a
/// `class_resolver` function must be provided.
///
/// See `torch::pickle` for details.
TORCH_API IValue unpickle(
    const char* data,
    size_t size,
    ClassResolver class_resolver = nullptr,
    const std::vector<at::Tensor>* tensor_table = nullptr);

} // namespace jit
} // namespace torch
