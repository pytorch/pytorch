#pragma once

#include <torch/csrc/python_headers.h>
#include <stdexcept>
#include <string>
#include <torch/csrc/utils/object_ptr.h>

// Utilities for handling Python strings. Note that PyString, when defined, is
// the same as PyBytes.

// Returns true if obj is a bytes/str or unicode object
// As of Python 3.6, this does not require the GIL
inline bool THPUtils_checkString(PyObject* obj) {
  return PyBytes_Check(obj) || PyUnicode_Check(obj);
}

// Unpacks PyBytes (PyString) or PyUnicode as std::string
// PyBytes are unpacked as-is. PyUnicode is unpacked as UTF-8.
// NOTE: this method requires the GIL
inline std::string THPUtils_unpackString(PyObject* obj) {
  if (PyBytes_Check(obj)) {
    size_t size = PyBytes_GET_SIZE(obj);
    return std::string(PyBytes_AS_STRING(obj), size);
  }
  if (PyUnicode_Check(obj)) {
    Py_ssize_t size;
    const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
    if (!data) {
      throw std::runtime_error("error unpacking string as utf-8");
    }
    return std::string(data, (size_t)size);
  }
  throw std::runtime_error("unpackString: expected bytes or unicode object");
}

inline PyObject* THPUtils_packString(const char* str) {
  return PyUnicode_FromString(str);
}

inline PyObject* THPUtils_packString(const std::string& str) {
  return PyUnicode_FromStringAndSize(str.c_str(), str.size());
}

inline PyObject* THPUtils_internString(const std::string& str) {
  return PyUnicode_InternFromString(str.c_str());
}

// Precondition: THPUtils_checkString(obj) must be true
inline bool THPUtils_isInterned(PyObject* obj) {
  return PyUnicode_CHECK_INTERNED(obj);
}

// Precondition: THPUtils_checkString(obj) must be true
inline void THPUtils_internStringInPlace(PyObject** obj) {
  PyUnicode_InternInPlace(obj);
}
