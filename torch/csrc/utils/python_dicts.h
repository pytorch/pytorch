#pragma once

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

inline PyObject* THPUtils_newDict() {
  PyObject* const dict = PyDict_New();
  if (dict == nullptr) {
    throw python_error();
  }
  return dict;
}

inline void THPUtils_setDictStrPyObject(PyObject* const dict, const char* key, PyObject* const value) {
  if (PyDict_SetItemString(dict, key, value) != 0) {
    throw python_error();
  }
}

inline void THPUtils_setDictStrStr(PyObject* const dict, const char* key, const char* value) {
  PyObject* const valObj = THPUtils_packString(value);
  if (valObj == nullptr) {
    throw python_error();
  }
  THPUtils_setDictStrPyObject(dict, key, valObj);
}

inline void THPUtils_setDictStrInt64(PyObject* const dict, const char* key, int64_t value) {
  PyObject* const valObj = THPUtils_packInt64(value);
  if (valObj == nullptr) {
    throw python_error();
  }
  THPUtils_setDictStrPyObject(dict, key, valObj);
}
