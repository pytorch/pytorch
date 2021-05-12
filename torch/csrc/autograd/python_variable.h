#pragma once

#include <torch/csrc/python_headers.h>
#include <memory>
#include <ATen/ATen.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/THP_export.h>

// Python object that backs torch.autograd.Variable
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPVariable {
    PyObject_HEAD
    // Payload
    torch::autograd::Variable cdata;
    // Hooks to be run on backwards pass (corresponds to Python attr
    // '_backwards_hooks', set by 'register_hook')
    PyObject* backward_hooks = nullptr;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
THP_API PyObject *THPVariableClass;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
THP_API PyObject *ParameterClass;

bool THPVariable_initModule(PyObject *module);
THP_API PyObject * THPVariable_Wrap(torch::autograd::Variable var);

static inline bool THPVariable_CheckTypeExact(PyTypeObject* tp) {
  // Check that a python object is a `Tensor`, but not a `Tensor` subclass.
  // (A subclass could have different semantics.) The one exception is
  // Parameter, which is used for Python bookkeeping but is equivalent to
  // Tensor as far as C++ is concerned.
  return (
    tp == (PyTypeObject*)THPVariableClass ||
    tp == (PyTypeObject*)ParameterClass
  );
}

static inline bool THPVariable_CheckExact(PyObject *obj) {
  return THPVariable_CheckTypeExact(Py_TYPE(obj));
}

inline bool THPVariable_Check(PyObject *obj)
{
  return THPVariableClass && PyObject_IsInstance(obj, THPVariableClass);
}

inline const at::Tensor& THPVariable_Unpack(THPVariable* var) {
  return var->cdata;
}

inline const at::Tensor& THPVariable_Unpack(PyObject* obj) {
  return THPVariable_Unpack(reinterpret_cast<THPVariable*>(obj));
}
