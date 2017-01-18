#pragma once

#include <Python.h>
#include <memory>

#include "torch/csrc/autograd/variable.h"

struct THPVariable {
    PyObject_HEAD
    std::shared_ptr<torch::autograd::Variable> cdata;
    PyObject* data;
    PyObject* backward_hooks;
};

bool THPVariable_initModule(PyObject *module);
extern PyObject *THPVariableClass;
PyObject * THPVariable_NewVolatile(PyObject *data);
PyObject * THPVariable_New(PyObject *data, PyObject *creator, bool requires_grad, bool is_volatile=false);
PyObject * THPVariable_Wrap(const std::shared_ptr<torch::autograd::Variable>& var);
PyObject * THPVariable_get_data(THPVariable *self);

inline bool THPVariable_Check(PyObject *obj)
{
  return THPVariableClass && PyObject_IsInstance(obj, THPVariableClass);
}
