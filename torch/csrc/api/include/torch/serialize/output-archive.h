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
class TORCH_API OutputArchive final {
 public:
  /// Default-constructs the `OutputArchive`.
  OutputArchive();

  // Move is allowed.
  OutputArchive(OutputArchive&&) = default;
  OutputArchive& operator=(OutputArchive&&) = default;

  // Copy is disallowed.
  OutputArchive(OutputArchive&) = delete;
  OutputArchive& operator=(OutputArchive&) = delete;

  /// Writes a `(key, tensor)` pair to the `OutputArchive`, and marks it as
  /// being or not being a buffer (non-differentiable tensor).
  void write(
      const std::string& key,
      const Tensor& tensor,
      bool is_buffer = false);

  /// Writes a nested `OutputArchive` under the given `key` to this
  /// `OutputArchive`.
  void write(const std::string& key, OutputArchive& nested_archive);

  /// Saves the `OutputArchive` into a serialized representation in a file at
  /// `filename`.
  void save_to(const std::string& filename);

  /// Saves the `OutputArchive` into a serialized representation into the given
  /// `stream`.
  void save_to(std::ostream& stream);

  /// Forwards all arguments to `write()`.
  /// Useful for generic code that can be re-used for both `OutputArchive` and
  /// `InputArchive` (where `operator()` forwards to `read()`).
  template <typename... Ts>
  void operator()(Ts&&... ts) {
    write(std::forward<Ts>(ts)...);
  }

 private:
  std::shared_ptr<jit::script::Module> module_;
};
} // namespace serialize
} // namespace torch
