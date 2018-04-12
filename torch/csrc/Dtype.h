#pragma once

#include <Python.h>
#include "ATen/ATen.h"

const int DTYPE_NAME_LEN = 64;

struct THPDtype {
  PyObject_HEAD
  at::ScalarType scalar_type;
  char name[DTYPE_NAME_LEN + 1];
};

extern PyTypeObject THPDtypeType;

inline bool THPDtype_Check(PyObject *obj) {
  return Py_TYPE(obj) == &THPDtypeType;
}

PyObject * THPDtype_New(at::ScalarType scalar_type, const std::string& name);

void THPDtype_init(PyObject *module);
