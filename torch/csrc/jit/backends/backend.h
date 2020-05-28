#pragma once

#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/backends/backend_interface.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/custom_class.h>

namespace torch {
namespace jit {

// Static registration API for backends.
template <class TBackendInterface>
class TORCH_API backend {
  static_assert(
      std::is_base_of<PyTorchBackendInterface, TBackendInterface>::value,
      "torch::jit::backend_<T> requires T to inherit from PyTorchBackendInterface");
  std::string backend_name_;

 public:
  explicit backend(const std::string& name) : backend_name_(name) {
    static auto cls =
        torch::class_<TBackendInterface>(detail::kBackendsNamespace, name)
            .def(torch::init<>())
            ._def_unboxed(
                "preprocess",
                detail::getPreprocessFunc<TBackendInterface>(),
                detail::getPreprocessSchema())
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

// Generates and returns a function that takes a Module and a lowering
// specification in the form of a dictionary. The caller is responsible for
// binding this into a CPython module.
std::function<Module(Module, py::dict)> generateToBackendFn(
    const std::string& backend_name);

} // namespace jit
} // namespace torch
