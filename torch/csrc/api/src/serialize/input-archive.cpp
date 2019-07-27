#include <torch/serialize/input-archive.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/script/module.h>

#include <c10/util/Exception.h>

#include <istream>
#include <memory>
#include <string>
#include <utility>

namespace torch {
namespace serialize {

InputArchive::InputArchive() {}

bool InputArchive::try_read(
    const std::string& key,
    Tensor& tensor,
    bool is_buffer) {
  auto param = module_.find_parameter(key);
  auto buffer = module_.find_buffer(key);
  if (!param && !buffer) return false;

  // clang-format off
  auto read_param = is_buffer ? buffer : param;
  auto read_tensor = read_param->value().toTensor();
  TORCH_CHECK(
      bool(buffer) == is_buffer,
      "Expected deserialized tensor for key '", key,
      "' to ", is_buffer ? "not " : "", "be a buffer, but it was not");
  // clang-format on
  if (tensor.defined()) {
    torch::NoGradGuard guard;
    if (tensor.device() != read_tensor.device()) {
      tensor.set_data(read_tensor);
    } else {
      tensor.set_(read_tensor);
    }
  } else {
    tensor = std::move(read_tensor);
  }
  return true;
}

void InputArchive::read(
    const std::string& key,
    Tensor& tensor,
    bool is_buffer) {
  TORCH_CHECK(
    try_read(key, tensor, is_buffer),
    "No such serialized tensor '",
    key,
    "'");
}

bool InputArchive::try_read(const std::string& key, InputArchive& archive) {
  if (auto named_module = module_.find_module(key)) {
    archive.module_ = std::move(*named_module);
    return true;
  } else {
    return false;
  }
}

void InputArchive::read(const std::string& key, InputArchive& archive) {
  TORCH_CHECK(
    try_read(key, archive),
    "No such serialized submodule: '", key, "'");
}

void InputArchive::load_from(const std::string& filename,
    c10::optional<torch::Device> device /*= c10::nullopt*/) {
  module_ = torch::jit::load(filename, std::move(device));
}

void InputArchive::load_from(std::istream& stream,
    c10::optional<torch::Device> device /*= c10::nullopt*/) {
  module_ = torch::jit::load(stream, std::move(device));
}
} // namespace serialize
} // namespace torch
