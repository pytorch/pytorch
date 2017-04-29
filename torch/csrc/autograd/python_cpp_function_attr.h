#pragma once

#include <Python.h>
#include "torch/csrc/autograd/python_cpp_function.h"

namespace torch { namespace autograd {

namespace attributes {

PyObject* next_functions(THPCppFunction* self, PyObject* hook);
PyObject* register_hook_dict(PyObject* self, PyObject* _var);
PyObject* register_hook(PyObject* self, PyObject* hook);

static struct PyMethodDef default_methods[] = {
  {(char*)"_register_hook_dict", (PyCFunction)register_hook_dict, METH_O, NULL},
  {(char*)"register_hook", (PyCFunction)register_hook, METH_O, NULL},
  {NULL}
};

static struct PyGetSetDef default_properties[] = {
  {(char*)"next_functions", (getter)next_functions, NULL, NULL, NULL},
  {NULL}
};

}

}}



