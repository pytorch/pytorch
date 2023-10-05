#pragma once

#include <stdexcept>

#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>
#include <string>

namespace torch::jit {

struct TORCH_API JITException : public std::runtime_error {
  explicit JITException(
      const std::string& msg,
      c10::optional<std::string> python_class_name = c10::nullopt,
      c10::optional<std::string> original_msg = c10::nullopt);

  c10::optional<std::string> getPythonClassName() const {
    return python_class_name_;
  }

  // the original msg if this is from a python exception. The interpretor has
  // changed the original message by adding "The following operation failed in
  // the TorchScript interpreter." in front of it in the handleError function.
  c10::optional<std::string> getOriginalMsg() const {
    return original_msg_;
  }

  static const std::string& getCaughtOriginalMsg();
  static const std::string& getCaughtPythonClassName();
  static void setCaughtOriginalMsg(const std::string& msg);
  static void setCaughtPythonClassName(const std::string& pythonClassName);

 private:
  c10::optional<std::string> python_class_name_;
  c10::optional<std::string> original_msg_;
};

} // namespace torch::jit
