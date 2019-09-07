#pragma once

#include <torch/csrc/python_headers.h>

#include <c10/core/QBackend.h>

#include <string>

constexpr int QBACKEND_NAME_LEN = 64;

struct THPQBackend {
  PyObject_HEAD at::QBackend qbackend;
  char name[QBACKEND_NAME_LEN + 1];
};

extern PyTypeObject THPQBackendType;

inline bool THPQBackend_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPQBackendType;
}

PyObject* THPQBackend_New(at::QBackend qbackend, const std::string& name);

void THPQBackend_init(PyObject* module);
