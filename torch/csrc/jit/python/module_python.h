#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/utils/pybind.h>
#include <tuple>

namespace py = pybind11;

namespace torch::jit {

inline std::optional<Module> as_module(py::handle obj) {
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  auto& ScriptModule =
      storage
          .call_once_and_store_result([]() -> py::object {
            return py::module_::import("torch.jit").attr("ScriptModule");
          })
          .get_stored();
  if (py::isinstance(obj, ScriptModule)) {
    return py::cast<Module>(obj.attr("_c"));
  }
  return std::nullopt;
}

inline std::optional<Object> as_object(py::handle obj) {
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<
      std::tuple<py::object, py::object>>
      storage;
  auto& [ScriptObject, RecursiveScriptClass] =
      storage
          .call_once_and_store_result(
              []() -> std::tuple<py::object, py::object> {
                return {
                    py::module_::import("torch").attr("ScriptObject"),
                    py::module_::import("torch.jit")
                        .attr("RecursiveScriptClass")};
              })
          .get_stored();
  if (py::isinstance(obj, ScriptObject)) {
    return py::cast<Object>(obj);
  }
  if (py::isinstance(obj, RecursiveScriptClass)) {
    return py::cast<Object>(obj.attr("_c"));
  }
  return std::nullopt;
}

} // namespace torch::jit
