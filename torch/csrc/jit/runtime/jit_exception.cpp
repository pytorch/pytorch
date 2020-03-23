#include <torch/csrc/jit/runtime/jit_exception.h>

namespace torch {
namespace jit {

JITException::JITException(const std::string& msg) : std::runtime_error(msg) {}

} // namespace jit
} // namespace torch
