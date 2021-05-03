#pragma once

#include <torch/csrc/python_headers.h>

#include <c10/core/MemoryFormat.h>

#include <string>

const int MEMORY_FORMAT_NAME_LEN = 64;

struct THPMemoryFormat {
  PyObject_HEAD
  at::MemoryFormat memory_format;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char name[MEMORY_FORMAT_NAME_LEN + 1];
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern PyTypeObject THPMemoryFormatType;

inline bool THPMemoryFormat_Check(PyObject *obj) {
  return Py_TYPE(obj) == &THPMemoryFormatType;
}

PyObject * THPMemoryFormat_New(at::MemoryFormat memory_format, const std::string& name);

void THPMemoryFormat_init(PyObject *module);
