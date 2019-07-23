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

inline bool THPUtils_checkU32String(PyObject* obj) {
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
#if PY_MAJOR_VERSION == 2
    THPObjectPtr bytes(PyUnicode_AsUTF8String(obj));
    if (!bytes) {
      throw std::runtime_error("error unpacking string as utf-8");
    }
    size_t size = PyBytes_GET_SIZE(bytes.get());
    return std::string(PyBytes_AS_STRING(bytes.get()), size);
#else
    Py_ssize_t size;
    const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
    if (!data) {
      throw std::runtime_error("error unpacking string as utf-8");
    }
    return std::string(data, (size_t)size);
#endif
  }
  throw std::runtime_error("unpackString: expected bytes or unicode object");
}

inline PyObject* THPUtils_packString(const char* str) {
#if PY_MAJOR_VERSION == 2
  return PyString_FromString(str);
#else
  return PyUnicode_FromString(str);
#endif
}

inline PyObject* THPUtils_packString(const std::string& str) {
#if PY_MAJOR_VERSION == 2
  return PyString_FromStringAndSize(str.c_str(), str.size());
#else
  return PyUnicode_FromStringAndSize(str.c_str(), str.size());
#endif
}

inline PyObject* THPUtils_internString(const std::string& str) {
#if PY_MAJOR_VERSION == 2
  return PyString_InternFromString(str.c_str());
#else
  return PyUnicode_InternFromString(str.c_str());
#endif
}

// Precondition: THPUtils_checkString(obj) must be true
inline bool THPUtils_isInterned(PyObject* obj) {
#if PY_MAJOR_VERSION == 2
  return PyString_CHECK_INTERNED(obj);
#else
  return PyUnicode_CHECK_INTERNED(obj);
#endif
}

// Precondition: THPUtils_checkString(obj) must be true
inline void THPUtils_internStringInPlace(PyObject** obj) {
#if PY_MAJOR_VERSION == 2
  PyString_InternInPlace(obj);
#else
  PyUnicode_InternInPlace(obj);
#endif
}

inline std::u32string THPUtils_unpackU32String(PyObject* obj) {
  if (PyBytes_Check(obj)) {
    size_t size = PyBytes_GET_SIZE(obj);
    if ((size & 3) != 0) {
      throw std::runtime_error("error unpacking string as utf-32: wrong size");
    }
    std::u32string str = std::u32string(
      reinterpret_cast<char32_t*>(PyBytes_AS_STRING(obj)), size / 4);
    str = str.erase(0, 1);      // remove the BOM mark
    return str;
  }
  if (PyUnicode_Check(obj)) {
    THPObjectPtr bytes(PyUnicode_AsUTF32String(obj));
    if (!bytes) {
      throw std::runtime_error("error unpacking string as utf-32");
    }
    size_t size = PyBytes_GET_SIZE(bytes.get());
    if ((size & 3) != 0) {
      throw std::runtime_error("error unpacking string as utf-32: wrong size");
    }
    std::u32string str = std::u32string(
      reinterpret_cast<char32_t*>(PyBytes_AS_STRING(bytes.get())), size / 4);
    str = str.erase(0, 1);      // remove the BOM mark
    return str;
  }
  throw std::runtime_error("unpackU32String: expected bytes or unicode object");
}
