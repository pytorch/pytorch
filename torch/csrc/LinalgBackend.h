#pragma once

#include <ATen/LinalgBackend.h>

#include <torch/csrc/python_headers.h>

#include <string>

const int LINALG_BACKEND_NAME_LEN = 64;

struct THPLinalgBackend {
  PyObject_HEAD
  at::LinalgBackend linalg_backend;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char name[LINALG_BACKEND_NAME_LEN + 1];
};

extern PyTypeObject THPLinalgBackendType;

inline bool THPLinalgBackend_Check(PyObject *obj) {
  return Py_TYPE(obj) == &THPLinalgBackendType;
}

PyObject * THPLinalgBackend_New(at::LinalgBackend linalg_backend);

void THPLinalgBackend_init(PyObject *module);
