#pragma once

#include <Python.h>
#include <stdint.h>
#include <stdexcept>
#include "torch/csrc/Exceptions.h"

// largest integer that can be represented consecutively in a double
const int64_t DOUBLE_INT_MAX = 9007199254740992;

inline PyObject* THPUtils_packInt64(int64_t value) {
#if PY_MAJOR_VERSION == 2
  if (sizeof(long) == sizeof(int64_t)) {
    return PyInt_FromLong(static_cast<long>(value));
  } else if (value <= INT32_MAX && value >= INT32_MIN) {
    return PyInt_FromLong(static_cast<long>(value));
  }
#endif
  return PyLong_FromLongLong(value);
}

inline PyObject* THPUtils_packUInt64(uint64_t value) {
#if PY_MAJOR_VERSION == 2
  if (value <= INT32_MAX) {
    return PyInt_FromLong(static_cast<long>(value));
  }
#endif
  return PyLong_FromUnsignedLongLong(value);
}

inline PyObject* THPUtils_packDoubleAsInt(double value) {
#if PY_MAJOR_VERSION == 2
  if (value <= INT32_MAX && value >= INT32_MIN) {
    return PyInt_FromLong(static_cast<long>(value));
  }
#endif
  return PyLong_FromDouble(value);
}

inline bool THPUtils_checkLong(PyObject* obj) {
#if PY_MAJOR_VERSION == 2
  return (PyLong_Check(obj) || PyInt_Check(obj)) && !PyBool_Check(obj);
#else
  return PyLong_Check(obj) && !PyBool_Check(obj);
#endif
}

inline int64_t THPUtils_unpackLong(PyObject* obj) {
  int overflow;
  long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  if (overflow != 0) {
    throw std::runtime_error("Overflow when unpacking long");
  }
  return (int64_t)value;
}

inline bool THPUtils_checkDouble(PyObject* obj) {
#if PY_MAJOR_VERSION == 2
  return PyFloat_Check(obj) || PyLong_Check(obj) || PyInt_Check(obj);
#else
  return PyFloat_Check(obj) || PyLong_Check(obj);
#endif
}

inline double THPUtils_unpackDouble(PyObject* obj) {
  if (PyFloat_Check(obj)) {
    return PyFloat_AS_DOUBLE(obj);
  }
  if (PyLong_Check(obj)) {
    int overflow;
    long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (overflow != 0) {
      throw std::runtime_error("Overflow when unpacking double");
    }
    if (value > DOUBLE_INT_MAX || value < -DOUBLE_INT_MAX) {
      throw std::runtime_error("Precision loss when unpacking double");
    }
    return (double)value;
  }
#if PY_MAJOR_VERSION == 2
  if (PyInt_Check(obj)) {
    return (double)PyInt_AS_LONG(obj);
  }
#endif
  double value = PyFloat_AsDouble(obj);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  return value;
}
