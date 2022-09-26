#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace torch {
namespace jit {

inline c10::optional<Module> as_module(py::handle obj) {
  static py::handle ScriptModule =
      py::module::import("torch.jit").attr("ScriptModule");
  if (py::isinstance(obj, ScriptModule)) {
    return py::cast<Module>(obj.attr("_c"));
  }
  return c10::nullopt;
}

inline c10::optional<Object> as_object(py::handle obj) {
  static py::handle ScriptObject =
      py::module::import("torch").attr("ScriptObject");
  if (py::isinstance(obj, ScriptObject)) {
    return py::cast<Object>(obj);
  }

  static py::handle RecursiveScriptClass =
      py::module::import("torch.jit").attr("RecursiveScriptClass");
  if (py::isinstance(obj, RecursiveScriptClass)) {
    return py::cast<Object>(obj.attr("_c"));
  }
  return c10::nullopt;
}

} // namespace jit
} // namespace torch
