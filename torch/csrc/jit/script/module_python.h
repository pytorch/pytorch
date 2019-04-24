#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/jit/script/module.h>

namespace py = pybind11;

namespace torch {
namespace jit {
namespace script {

inline std::shared_ptr<Module> as_module(const py::object& obj) {
  if (py::isinstance(
          obj, py::module::import("torch.jit").attr("ScriptModule"))) {
    return py::cast<std::shared_ptr<Module>>(obj.attr("_c"));
  }
  return nullptr;
}

} // namespace script
} // namespace jit
} // namespace torch
