#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/python_headers.h>

#include <ATen/ATen.h>

struct THPDTypeInfo {
  PyObject_HEAD
  at::ScalarType type;
};

struct THPFInfo : THPDTypeInfo {};

struct THPIInfo : THPDTypeInfo {};

TORCH_PYTHON_API extern PyTypeObject THPFInfoType;
TORCH_PYTHON_API extern PyTypeObject THPIInfoType;

inline bool THPFInfo_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPFInfoType;
}

inline bool THPIInfo_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPIInfoType;
}

void THPDTypeInfo_init(PyObject* module);
