#include <torch/csrc/jit/runtime/jit_exception.h>

namespace torch {
namespace jit {

JITException::JITException(
    const std::string& msg,
    c10::optional<std::string> python_class_name)
    : std::runtime_error(msg),
      python_class_name_(std::move(python_class_name)) {}

} // namespace jit
} // namespace torch
