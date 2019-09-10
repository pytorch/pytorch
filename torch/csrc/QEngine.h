#pragma once

#include <torch/csrc/python_headers.h>

#include <c10/core/QEngine.h>

#include <string>

constexpr int QENGINE_NAME_LEN = 64;

struct THPQEngine {
  PyObject_HEAD at::QEngine qengine;
  char name[QENGINE_NAME_LEN + 1];
};

extern PyTypeObject THPQEngineType;

inline bool THPQEngine_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPQEngineType;
}

PyObject* THPQEngine_New(at::QEngine qengine, const std::string& name);

void THPQEngine_init(PyObject* module);
