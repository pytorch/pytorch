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

THP_API PyObject *THPVariableClass;

bool THPVariable_initModule(PyObject *module);

// Creates a new Python object for a Variable. The Variable must not already
// have a PyObject* associated with it.
static inline PyObject * THPVariable_NewWithVar(PyTypeObject* type, torch::autograd::Variable var) {
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*) obj;
    new (&v->cdata) torch::autograd::Variable(std::move(var));
    torch::autograd::impl::set_pyobj(v->cdata, obj);
  }
  return obj;
}

THP_API PyObject * THPVariable_Wrap(torch::autograd::Variable var);

static inline bool THPVariable_CheckExact(PyObject *obj) {
  return Py_TYPE(obj) == (PyTypeObject*)THPVariableClass;
}

inline bool THPVariable_Check(PyObject *obj)
{
  return THPVariableClass && PyObject_IsInstance(obj, THPVariableClass);
}

inline torch::autograd::Variable& THPVariable_Unpack(PyObject* obj) {
  auto var = (THPVariable*)obj;
  return var->cdata;
}
