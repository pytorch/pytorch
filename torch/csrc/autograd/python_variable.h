#pragma once

#include <Python.h>
#include <memory>
#include <ATen/ATen.h>

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/THP_export.h"

// Python object that backs torch.autograd.Variable
struct THPVariable {
    PyObject_HEAD
    // Payload
    torch::autograd::Variable cdata;
    // Tensor this wraps (corresponds to Python attr 'data').
    // It assumed that a THPVariable is *uniquely* identified by the
    // tensor it wraps.
    // Invariant: v->data == v->cdata->data
    PyObject* data;
    // Hooks to be run on backwards pass (corresponds to Python attr
    // '_backwards_hooks', set by 'register_hook')
    PyObject* backward_hooks;
};

THP_API PyObject *THPVariableClass;

bool THPVariable_initModule(PyObject *module);
PyObject * THPVariable_Wrap(torch::autograd::Variable var);
PyObject * THPVariable_get_data(THPVariable *self);

inline bool THPVariable_Check(PyObject *obj)
{
  return THPVariableClass && PyObject_IsInstance(obj, THPVariableClass);
}
