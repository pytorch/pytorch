#pragma once

#include <stdexcept>

#include <torch/csrc/Export.h>
#include <string>

namespace torch {
namespace jit {

struct TORCH_API JITException : public std::runtime_error {
  explicit JITException(const std::string& msg);
};

struct TORCH_API CustomJITException : public JITException {
  explicit CustomJITException(
      const std::string& msg,
      std::string python_class_name);

  const std::string& getPythonClassName() const {
    return python_class_name_;
  }

 private:
  std::string python_class_name_;
};

} // namespace jit
} // namespace torch
