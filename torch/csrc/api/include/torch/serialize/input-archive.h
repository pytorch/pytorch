#pragma once

#include <c10/core/Device.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/types.h>
#include <optional>

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
struct Module;
} // namespace jit
} // namespace torch

namespace torch::serialize {

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

  /// Reads an `IValue` associated with a given `key`. If there is no `IValue`
  /// associated with the `key`, this returns false, otherwise it returns true.
  bool try_read(const std::string& key, c10::IValue& ivalue);

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
  void load_from(
      const std::string& filename,
      std::optional<torch::Device> device = std::nullopt);

  /// Loads the `InputArchive` from a serialized representation stored in the
  /// given `stream`. Storage are remapped using device option. If device
  /// is not specified, the module is loaded to the original device.
  void load_from(
      std::istream& stream,
      std::optional<torch::Device> device = std::nullopt);

  // Loads given the specified flat array.
  void load_from(
      const char* data,
      size_t size,
      std::optional<torch::Device> device = std::nullopt);

  // Loads given the specified read and size functions.
  void load_from(
      const std::function<size_t(uint64_t pos, void* buf, size_t nbytes)>&
          read_func,
      const std::function<size_t(void)>& size_func,
      std::optional<torch::Device> device = std::nullopt);

  // Returns the vector of keys in the input archive.
  std::vector<std::string> keys();

  /// Forwards all arguments to `read()`.
  /// Useful for generic code that can be re-used for both `InputArchive` and
  /// `OutputArchive` (where `operator()` forwards to `write()`).
  template <typename... Ts>
  void operator()(Ts&&... ts) {
    read(std::forward<Ts>(ts)...);
  }

 private:
  jit::Module module_;
  std::string hierarchy_prefix_;
};
} // namespace torch::serialize
