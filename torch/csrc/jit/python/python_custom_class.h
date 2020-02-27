
copy: fbcode/caffe2/torch/csrc/jit/python/python_custom_class.h
copyrev: 92b3b24c3304ddf028a3c79bfc60fcc0967129b9

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
