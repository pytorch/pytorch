#pragma once
#include <torch/csrc/Export.h>
#include <torch/csrc/python_headers.h>

#include <ATen/Layout.h>

#include <string>

const int LAYOUT_NAME_LEN = 64;

struct THPLayout {
  PyObject_HEAD
  at::Layout layout;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char name[LAYOUT_NAME_LEN + 1];
};

TORCH_PYTHON_API extern PyTypeObject THPLayoutType;

inline bool THPLayout_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPLayoutType;
}

PyObject* THPLayout_New(at::Layout layout, const std::string& name);

void THPLayout_init(PyObject* module);
