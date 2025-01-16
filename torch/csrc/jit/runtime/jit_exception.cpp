#include <torch/csrc/jit/runtime/jit_exception.h>

namespace torch::jit {

static thread_local std::string caughtOriginalMsg = "";
static thread_local std::string caughtPythonClassName = "";

JITException::JITException(
    const std::string& msg,
    std::optional<std::string> python_class_name,
    std::optional<std::string> original_msg)
    : std::runtime_error(msg),
      python_class_name_(std::move(python_class_name)),
      original_msg_(std::move(original_msg)) {}

const std::string& JITException::getCaughtOriginalMsg() {
  return caughtOriginalMsg;
}
const std::string& JITException::getCaughtPythonClassName() {
  return caughtPythonClassName;
}
void JITException::setCaughtOriginalMsg(const std::string& msg) {
  caughtOriginalMsg = msg;
}
void JITException::setCaughtPythonClassName(
    const std::string& pythonClassName) {
  caughtPythonClassName = pythonClassName;
}

} // namespace torch::jit
