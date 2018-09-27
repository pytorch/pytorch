#include <torch/serialize/input-archive.h>

#include <torch/tensor.h>
#include <torch/utils.h>

#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/script/module.h>

#include <ATen/core/Error.h>

#include <memory>
#include <string>
#include <utility>

namespace torch {
namespace serialize {

InputArchive::InputArchive()
    : module_(std::make_shared<jit::script::Module>()) {}

void InputArchive::read(
    const std::string& key,
    Tensor& tensor,
    bool is_buffer) {
  auto* read_tensor = module_->find_parameter(key);
  AT_CHECK(read_tensor != nullptr, "No such serialized tensor '", key, "'");
  // clang-format off
  AT_CHECK(
      read_tensor->is_buffer == is_buffer,
      "Expected deserialized tensor for key '", key,
      "' to ", is_buffer ? "not " : "", "be a buffer, but it was not");
  // clang-format on
  if (tensor.defined()) {
    torch::NoGradGuard guard;
    tensor.set_(*read_tensor->slot());
  } else {
    tensor = std::move(*read_tensor->slot());
  }
}

void InputArchive::read(const std::string& key, InputArchive& archive) {
  if (auto* named_module = module_->find_module(key)) {
    AT_ASSERT(named_module->module != nullptr);
    archive.module_ = std::move(named_module->module);
  } else {
    AT_ERROR("No such serialized submodule: '", key, "'");
  }
}

InputArchive::InputArchive(std::shared_ptr<jit::script::Module> module)
    : module_(std::move(module)) {}

InputArchive load_from_file(const std::string& filename) {
  return InputArchive(torch::jit::load(filename));
}
} // namespace serialize
} // namespace torch
