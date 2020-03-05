#pragma once

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/custom_class.h>

namespace torch {
namespace jit {

void initPythonCustomClassBindings(PyObject* module);

struct ScriptClass {
  ScriptClass(c10::StrongTypePtr class_type)
      : class_type_(std::move(class_type)) {}

  py::object __call__(py::args args, py::kwargs kwargs);

  c10::StrongTypePtr class_type_;
};

} // namespace jit
} // namespace torch
