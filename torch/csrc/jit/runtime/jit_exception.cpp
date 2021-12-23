#include <torch/csrc/jit/runtime/jit_exception.h>

namespace torch {
namespace jit {

JITException::JITException(const std::string& msg) : std::runtime_error(msg) {}

CustomJITException::CustomJITException(
    const std::string& msg,
    std::string python_class_name)
    : JITException(msg), python_class_name_(std::move(python_class_name)) {}

} // namespace jit
} // namespace torch
