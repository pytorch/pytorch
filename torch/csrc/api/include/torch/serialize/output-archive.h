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
class OutputArchive final {
 public:
  OutputArchive();

  void write(
      const std::string& key,
      const Tensor& tensor,
      bool is_buffer = false);
  void write(const std::string& key, OutputArchive& nested_archive);

  template <typename... Ts>
  void operator()(Ts&&... ts) {
    write(std::forward<Ts>(ts)...);
  }

 private:
  friend void save_to_file(const OutputArchive&, const std::string&);

  std::shared_ptr<jit::script::Module> module_;
};

void save_to_file(const OutputArchive& archive, const std::string& filename);
} // namespace serialize
} // namespace torch
