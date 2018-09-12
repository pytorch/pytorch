#include <torch/serialize/default.h>

#include <torch/serialize/base.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <torch/csrc/jit/export.h>
#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/script/module.h>

#include <memory>
#include <string>

namespace torch {
namespace serialize {

namespace detail {
inline std::string replace_dots_with_underscores(std::string string) {
  for (char& c : string) {
    if (c == '.') {
      c = '_';
    }
  }
  return string;
}
} // namespace detail

DefaultWriter::DefaultWriter(std::string filename) : filename_(filename) {}

void DefaultWriter::write(
    const std::string& key,
    const Tensor& tensor,
    bool is_buffer) {
  auto patched_key = detail::replace_dots_with_underscores(key);
  module_.register_parameter(patched_key, tensor, is_buffer);
}

void DefaultWriter::finish() {
  jit::ExportModule(module_, filename_);
}

DefaultReader::DefaultReader(std::string filename)
    : module_(jit::load(filename)) {}

void DefaultReader::read(
    const std::string& key,
    Tensor& tensor,
    bool is_buffer) {
  auto patched_key = detail::replace_dots_with_underscores(key);
  auto* read_tensor = module_->find_parameter(patched_key);
  AT_CHECK(read_tensor != nullptr, "No such serialized tensor '", key, "'");
  AT_CHECK(
      read_tensor->is_buffer == is_buffer,
      "Expected deserialized tensor for key '",
      key,
      "' to ",
      is_buffer ? "not " : "",
      "be a buffer, but it was not");
  if (tensor.defined()) {
    torch::NoGradGuard guard;
    tensor.set_(*read_tensor->slot());
  } else {
    tensor = std::move(*read_tensor->slot());
  }
}
} // namespace serialize
} // namespace torch
