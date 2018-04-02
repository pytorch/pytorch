#pragma once

#include <Python.h>
#include <string>

const int LAYOUT_NAME_LEN = 64;

struct THPLayout {
  PyObject_HEAD
  bool is_strided;
  char name[LAYOUT_NAME_LEN + 1];
};

extern PyTypeObject THPLayoutType;

inline bool THPLayout_Check(PyObject *obj) {
  return Py_TYPE(obj) == &THPLayoutType;
}

PyObject * THPLayout_New(bool is_strided, const std::string& name);

void THPLayout_init(PyObject *module);
