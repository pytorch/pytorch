#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <stdexcept>
#include <string>

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
    Py_ssize_t size = 0;
    const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
    if (!data) {
      throw std::runtime_error("error unpacking string as utf-8");
    }
    return std::string(data, (size_t)size);
  }
  throw std::runtime_error("unpackString: expected bytes or unicode object");
}

// Unpacks PyBytes (PyString) or PyUnicode as std::string_view
// PyBytes are unpacked as-is. PyUnicode is unpacked as UTF-8.
// NOTE: If `obj` is destroyed, then the non-owning std::string_view will
//   become invalid. If the string needs to be accessed at any point after
//   `obj` is destroyed, then the std::string_view should be copied into
//   a std::string, or another owning object, and kept alive. For an example,
//   look at how IValue and autograd nodes handle std::string_view arguments.
// NOTE: this method requires the GIL
inline std::string_view THPUtils_unpackStringView(PyObject* obj) {
  if (PyBytes_Check(obj)) {
    size_t size = PyBytes_GET_SIZE(obj);
    return std::string_view(PyBytes_AS_STRING(obj), size);
  }
  if (PyUnicode_Check(obj)) {
    Py_ssize_t size = 0;
    const char* data = PyUnicode_AsUTF8AndSize(obj, &size);
    if (!data) {
      throw std::runtime_error("error unpacking string as utf-8");
    }
    return std::string_view(data, (size_t)size);
  }
  throw std::runtime_error("unpackString: expected bytes or unicode object");
}

inline PyObject* THPUtils_packString(const char* str) {
  return PyUnicode_FromString(str);
}

inline PyObject* THPUtils_packString(const std::string& str) {
  return PyUnicode_FromStringAndSize(
      str.c_str(), static_cast<Py_ssize_t>(str.size()));
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

/*
 * Reference:
 * https://github.com/numpy/numpy/blob/f4c497c768e0646df740b647782df463825bfd27/numpy/core/src/common/get_attr_string.h#L42
 *
 * Stripped down version of PyObject_GetAttrString,
 * avoids lookups for None, tuple, and List objects,
 * and doesn't create a PyErr since this code ignores it.
 *
 * This can be much faster then PyObject_GetAttrString where
 * exceptions are not used by caller.
 *
 * 'obj' is the object to search for attribute.
 *
 * 'name' is the attribute to search for.
 *
 * Returns a py::object wrapping the return value. If the attribute lookup
 * failed the value will be NULL.
 *
 */

inline py::object PyObject_FastGetAttrString(PyObject* obj, const char* name) {
  PyTypeObject* tp = Py_TYPE(obj);
  PyObject* res = (PyObject*)nullptr;

  /* Attribute referenced by (char *)name */
  if (tp->tp_getattr != nullptr) {
    // This is OK per https://bugs.python.org/issue39620
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    res = (*tp->tp_getattr)(obj, const_cast<char*>(name));
    if (res == nullptr) {
      PyErr_Clear();
    }
  }
  /* Attribute referenced by (PyObject *)name */
  else if (tp->tp_getattro != nullptr) {
    auto w = py::reinterpret_steal<py::object>(PyUnicode_FromString(name));
    if (w.ptr() == nullptr) {
      return py::object();
    }
    res = (*tp->tp_getattro)(obj, w.ptr());
    if (res == nullptr) {
      PyErr_Clear();
    }
  }
  return py::reinterpret_steal<py::object>(res);
}
