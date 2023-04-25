#pragma once

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <cstdint>
#include <limits>
#include <stdexcept>

// largest integer that can be represented consecutively in a double
const int64_t DOUBLE_INT_MAX = 9007199254740992;

inline PyObject* THPUtils_packInt32(int32_t value) {
  return PyLong_FromLong(value);
}

inline PyObject* THPUtils_packInt64(int64_t value) {
  return PyLong_FromLongLong(value);
}

inline PyObject* THPUtils_packUInt32(uint32_t value) {
  return PyLong_FromUnsignedLong(value);
}

inline PyObject* THPUtils_packUInt64(uint64_t value) {
  return PyLong_FromUnsignedLongLong(value);
}

inline PyObject* THPUtils_packDoubleAsInt(double value) {
  return PyLong_FromDouble(value);
}

inline bool THPUtils_checkLongExact(PyObject* obj) {
  return PyLong_CheckExact(obj) && !PyBool_Check(obj);
}

inline bool THPUtils_checkLong(PyObject* obj) {
  // Fast path
  if (THPUtils_checkLongExact(obj)) {
    return true;
  }

#ifdef USE_NUMPY
  if (torch::utils::is_numpy_int(obj)) {
    return true;
  }
#endif

  return PyLong_Check(obj) && !PyBool_Check(obj);
}

inline int32_t THPUtils_unpackInt(PyObject* obj) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int overflow;
  long value = PyLong_AsLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  if (overflow != 0) {
    throw std::runtime_error("Overflow when unpacking long");
  }
  if (value > std::numeric_limits<int32_t>::max() ||
      value < std::numeric_limits<int32_t>::min()) {
    throw std::runtime_error("Overflow when unpacking long");
  }
  return (int32_t)value;
}

inline int64_t THPUtils_unpackLong(PyObject* obj) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
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

inline uint32_t THPUtils_unpackUInt32(PyObject* obj) {
  unsigned long value = PyLong_AsUnsignedLong(obj);
  if (PyErr_Occurred()) {
    throw python_error();
  }
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Overflow when unpacking unsigned long");
  }
  return (uint32_t)value;
}

inline uint64_t THPUtils_unpackUInt64(PyObject* obj) {
  unsigned long long value = PyLong_AsUnsignedLongLong(obj);
  if (PyErr_Occurred()) {
    throw python_error();
  }
  return (uint64_t)value;
}

bool THPUtils_checkIndex(PyObject* obj);

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

inline bool THPUtils_checkBool(PyObject* obj) {
#ifdef USE_NUMPY
  if (torch::utils::is_numpy_bool(obj)) {
    return true;
  }
#endif
  return PyBool_Check(obj);
}

inline bool THPUtils_checkDouble(PyObject* obj) {
#ifdef USE_NUMPY
  if (torch::utils::is_numpy_scalar(obj)) {
    return true;
  }
#endif
  return PyFloat_Check(obj) || PyLong_Check(obj);
}

inline double THPUtils_unpackDouble(PyObject* obj) {
  if (PyFloat_Check(obj)) {
    return PyFloat_AS_DOUBLE(obj);
  }
  double value = PyFloat_AsDouble(obj);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  return value;
}

inline c10::complex<double> THPUtils_unpackComplexDouble(PyObject* obj) {
  Py_complex value = PyComplex_AsCComplex(obj);
  if (value.real == -1.0 && PyErr_Occurred()) {
    throw python_error();
  }

  return c10::complex<double>(value.real, value.imag);
}

inline bool THPUtils_unpackNumberAsBool(PyObject* obj) {
  if (PyFloat_Check(obj)) {
    return (bool)PyFloat_AS_DOUBLE(obj);
  }

  if (PyComplex_Check(obj)) {
    double real_val = PyComplex_RealAsDouble(obj);
    double imag_val = PyComplex_ImagAsDouble(obj);
    return !(real_val == 0 && imag_val == 0);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int overflow;
  long long value = PyLong_AsLongLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  // No need to check overflow, because when overflow occured, it should
  // return true in order to keep the same behavior of numpy.
  return (bool)value;
}
