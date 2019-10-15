#pragma once

#include <c10/util/Optional.h>
#include <c10/core/Device.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>
#include <torch/csrc/jit/script/module.h>

#include <iosfwd>
#include <memory>
#include <string>
#include <utility>

namespace at {
class Tensor;
} // namespace at

namespace torch {
using at::Tensor;
namespace jit {
namespace script {
struct Module;
} // namespace script
} // namespace jit
} // namespace torch

namespace torch {
namespace serialize {

/// A recursive representation of tensors that can be deserialized from a file
/// or stream. In most cases, users should not have to interact with this class,
/// and should instead use `torch::load`.
class TORCH_API InputArchive final {
 public:
  /// Default-constructs the `InputArchive`.
  InputArchive();

  // Move is allowed.
  InputArchive(InputArchive&&) = default;
  InputArchive& operator=(InputArchive&&) = default;

  // Copy is disallowed.
  InputArchive(InputArchive&) = delete;
  InputArchive& operator=(InputArchive&) = delete;

  ~InputArchive() = default;

  /// Reads an `IValue` associated with a given `key`.
  void read(const std::string& key, c10::IValue& ivalue);

  /// Reads a `tensor` associated with a given `key`. If there is no `tensor`
  /// associated with the `key`, this returns false, otherwise it returns true.
  /// If the tensor is expected to be a buffer (not differentiable), `is_buffer`
  /// must be `true`.
  bool try_read(const std::string& key, Tensor& tensor, bool is_buffer = false);

  /// Reads a `tensor` associated with a given `key`.
  /// If the tensor is expected to be a buffer (not differentiable), `is_buffer`
  /// must be `true`.
  void read(const std::string& key, Tensor& tensor, bool is_buffer = false);

  /// Reads a `InputArchive` associated with a given `key`. If there is no
  /// `InputArchive` associated with the `key`, this returns false, otherwise
  /// it returns true.
  bool try_read(const std::string& key, InputArchive& archive);

  /// Reads an `InputArchive` associated with a given `key`.
  /// The archive can thereafter be used for further deserialization of the
  /// nested data.
  void read(const std::string& key, InputArchive& archive);

  /// Loads the `InputArchive` from a serialized representation stored in the
  /// file at `filename`. Storage are remapped using device option. If device
  /// is not specified, the module is loaded to the original device.
  void load_from(const std::string& filename,
      c10::optional<torch::Device> device = c10::nullopt);

  /// Loads the `InputArchive` from a serialized representation stored in the
  /// given `stream`. Storage are remapped using device option. If device
  /// is not specified, the module is loaded to the original device.
  void load_from(std::istream& stream,
      c10::optional<torch::Device> device = c10::nullopt);

  /// Forwards all arguments to `read()`.
  /// Useful for generic code that can be re-used for both `InputArchive` and
  /// `OutputArchive` (where `operator()` forwards to `write()`).
  template <typename... Ts>
  void operator()(Ts&&... ts) {
    read(std::forward<Ts>(ts)...);
  }

 private:
  jit::script::Module module_;
};
} // namespace serialize
} // namespace torch
