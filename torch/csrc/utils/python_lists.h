#pragma once

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>

inline PyObject* THPUtils_newList(size_t size = 0) {
  PyObject* const list = PyList_New(size);
  if (list == nullptr) {
    throw python_error();
  }
  return list;
}

inline void THPUtils_appendListPyObject(PyObject* const list, PyObject* const value) {
  if (PyList_Append(list, value) != 0) {
    throw python_error();
  }
}
