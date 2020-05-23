#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

const int DTYPE_NAME_LEN = 64;

struct TORCH_API THPDtype {
  PyObject_HEAD
  at::ScalarType scalar_type;
  char name[DTYPE_NAME_LEN + 1];
};

TORCH_API extern PyTypeObject THPDtypeType;

inline bool THPDtype_Check(PyObject *obj) {
  return Py_TYPE(obj) == &THPDtypeType;
}

inline bool THPPythonScalarType_Check(PyObject *obj) {
  return obj == (PyObject*)(&PyFloat_Type) ||
    obj == (PyObject*)(&PyBool_Type) ||
    obj == (PyObject*)(&PyLong_Type);
}

PyObject * THPDtype_New(at::ScalarType scalar_type, const std::string& name);

void THPDtype_init(PyObject *module);
