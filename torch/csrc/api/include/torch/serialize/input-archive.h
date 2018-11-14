#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

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

  /// Reads a `tensor` associated with a given `key`.
  /// If the tensor is expected to be a buffer (not differentiable), `is_buffer`
  /// must be `true`.
  void read(const std::string& key, Tensor& tensor, bool is_buffer = false);

  /// Reads an `InputArchive` associated with a given `key`.
  /// The archive can thereafter be used for further deserialization of the
  /// nested data.
  void read(const std::string& key, InputArchive& archive);

  /// Loads the `InputArchive` from a serialized representation stored in the
  /// file at `filename`.
  void load_from(const std::string& filename);

  /// Loads the `InputArchive` from a serialized representation stored in the
  /// given `stream`.
  void load_from(std::istream& stream);

  /// Forwards all arguments to `read()`.
  /// Useful for generic code that can be re-used for both `InputArchive` and
  /// `OutputArchive` (where `operator()` forwards to `write()`).
  template <typename... Ts>
  void operator()(Ts&&... ts) {
    read(std::forward<Ts>(ts)...);
  }

 private:
  std::shared_ptr<jit::script::Module> module_;
};
} // namespace serialize
} // namespace torch
