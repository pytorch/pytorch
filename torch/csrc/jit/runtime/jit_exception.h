#pragma once

#include <stdexcept>

#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>
#include <string>

namespace torch {
namespace jit {

struct TORCH_API JITException : public std::runtime_error {
  explicit JITException(
      const std::string& msg,
      c10::optional<std::string> python_class_name = c10::nullopt);

  c10::optional<std::string> getPythonClassName() const {
    return python_class_name_;
  }

 private:
  c10::optional<std::string> python_class_name_;
};

} // namespace jit
} // namespace torch
