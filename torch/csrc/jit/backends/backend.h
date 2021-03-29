#pragma once

#include <ATen/core/builtin_function.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/backends/backend_interface.h>
#include <torch/custom_class.h>

namespace torch {
namespace jit {

// Static registration API for backends.
template <class TBackendInterface>
class backend {
  static_assert(
      std::is_base_of<PyTorchBackendInterface, TBackendInterface>::value,
      "torch::jit::backend<T> requires T to inherit from PyTorchBackendInterface");
  std::string backend_name_;

 public:
  // Registers a new backend with /p name, and the given /p preprocess
  // function.
  backend(
      const std::string& name,
      const detail::BackendPreprocessFunction& preprocess)
      : backend_name_(name) {
    detail::registerBackendPreprocessFunction(name, preprocess);
    static auto cls =
        torch::class_<TBackendInterface>(detail::kBackendsNamespace, name)
            .def(torch::init<>())
            ._def_unboxed(
                "is_available",
                detail::getIsAvailableFunc<TBackendInterface>(),
                detail::getIsAvailableSchema())
            ._def_unboxed(
                "compile",
                detail::getCompileFunc<TBackendInterface>(),
                detail::getCompileSchema())
            ._def_unboxed(
                "execute",
                detail::getExecuteFunc<TBackendInterface>(),
                detail::getExecuteSchema());
  }
};

} // namespace jit
} // namespace torch
