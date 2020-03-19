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

// Given a qualified name (e.g. __torch__.torch.classes.Foo), return
// the ClassType pointer to the Type that describes that custom class,
// or nullptr if no class by that name was found.
TORCH_API at::ClassTypePtr getCustomClass(const std::string& name);

// Given an IValue, return true if the object contained in that IValue
// is a custom C++ class, otherwise return false.
TORCH_API bool isCustomClass(const c10::IValue& v);

} // namespace jit
} // namespace torch
