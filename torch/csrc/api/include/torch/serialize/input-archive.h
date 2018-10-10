#pragma once

#include <torch/tensor.h>

#include <memory>
#include <string>
#include <utility>

namespace torch {
namespace jit {
namespace script {
struct Module;
} // namespace script
} // namespace jit
} // namespace torch

namespace torch {
namespace serialize {
class InputArchive final {
 public:
  InputArchive();

  void read(const std::string& key, Tensor& tensor, bool is_buffer = false);
  void read(const std::string& key, InputArchive& archive);

  template <typename... Ts>
  void operator()(Ts&&... ts) {
    read(std::forward<Ts>(ts)...);
  }

 private:
  friend InputArchive load_from_file(const std::string& filename);

  InputArchive(std::shared_ptr<jit::script::Module> module);

  std::shared_ptr<jit::script::Module> module_;
};

InputArchive load_from_file(const std::string& filename);
} // namespace serialize
} // namespace torch
