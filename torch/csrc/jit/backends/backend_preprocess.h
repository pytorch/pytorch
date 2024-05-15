#pragma once

#include <torch/csrc/jit/backends/backend_detail.h>
namespace torch {
namespace jit {
class backend_preprocess_register {
  std::string backend_name_;

 public:
  backend_preprocess_register(
      const std::string& name,
      const detail::BackendPreprocessFunction& preprocess)
      : backend_name_(name) {
    detail::registerBackendPreprocessFunction(name, preprocess);
  }
};
} // namespace jit
} // namespace torch
