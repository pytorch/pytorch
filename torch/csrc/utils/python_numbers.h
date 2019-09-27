#pragma once

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <cstdint>
#include <stdexcept>

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

inline bool THPUtils_checkIndex(PyObject *obj) {
  if (PyBool_Check(obj)) {
    return false;
  }
  if (THPUtils_checkLong(obj)) {
    return true;
  }
  torch::jit::tracer::NoWarn no_warn_guard;
  auto index = THPObjectPtr(PyNumber_Index(obj));
  if (!index) {
    PyErr_Clear();
    return false;
  }
  return true;
}

inline int64_t THPUtils_unpackIndex(PyObject* obj) {
  if (!THPUtils_checkLong(obj)) {
    auto index = THPObjectPtr(PyNumber_Index(obj));
    if (index == nullptr) {
      throw python_error();
    }
    // NB: This needs to be called before `index` goes out of scope and the
    // underlying object's refcount is decremented
    return THPUtils_unpackLong(index.get());
  }
  return THPUtils_unpackLong(obj);
}

inline bool THPUtils_unpackBool(PyObject* obj) {
  if (obj == Py_True) {
    return true;
  } else if (obj == Py_False) {
    return false;
  } else {
    throw std::runtime_error("couldn't convert python object to boolean");
  }
}

inline bool THPUtils_checkDouble(PyObject* obj) {
  bool is_numpy_scalar;
#ifdef USE_NUMPY
  is_numpy_scalar = torch::utils::is_numpy_scalar(obj);
#else
  is_numpy_scalar = false;
#endif
#if PY_MAJOR_VERSION == 2
  return PyFloat_Check(obj) || PyLong_Check(obj) || PyInt_Check(obj) || is_numpy_scalar;
#else
  return PyFloat_Check(obj) || PyLong_Check(obj) || is_numpy_scalar;
#endif
}

inline bool THPUtils_checkScalar(PyObject* obj) {
  bool is_numpy_scalar;
#ifdef USE_NUMPY
  is_numpy_scalar = torch::utils::is_numpy_scalar(obj);
#else
  is_numpy_scalar = false;
#endif
#if PY_MAJOR_VERSION == 2
  return PyFloat_Check(obj) || PyLong_Check(obj) || PyInt_Check(obj) || PyComplex_Check(obj) || is_numpy_scalar;
#else
  return PyFloat_Check(obj) || PyLong_Check(obj) || PyComplex_Check(obj) || is_numpy_scalar;
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

inline std::complex<double> THPUtils_unpackComplexDouble(PyObject *obj) {
  Py_complex value = PyComplex_AsCComplex(obj);
  if (value.real == -1.0 && PyErr_Occurred()) {
    throw python_error();
  }

  return std::complex<double>(value.real, value.imag);
}
