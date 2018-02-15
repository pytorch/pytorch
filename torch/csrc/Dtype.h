#pragma once

#include <Python.h>
#include "ATen/ATen.h"

const int DTYPE_NAME_LEN = 64;

struct THPDtype {
  PyObject_HEAD
  at::Type *cdata;
  char name[DTYPE_NAME_LEN + 1];
  bool is_cuda;
  bool is_sparse;
};

extern PyObject *THPDtypeClass;

inline bool THPDtype_Check(PyObject *obj) {
  return (PyObject*)Py_TYPE(obj) == THPDtypeClass;
}

PyObject * THPDtype_New(at::Type* cdata, const std::string& name, bool is_cuda, bool is_sparse);

bool THPDtype_init(PyObject *module);
