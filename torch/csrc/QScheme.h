#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/python_headers.h>

#include <c10/core/QScheme.h>

#include <string>

constexpr int QSCHEME_NAME_LEN = 64;

struct THPQScheme {
  PyObject_HEAD
  at::QScheme qscheme;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char name[QSCHEME_NAME_LEN + 1];
};

TORCH_PYTHON_API extern PyTypeObject THPQSchemeType;

inline bool THPQScheme_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPQSchemeType;
}

PyObject* THPQScheme_New(at::QScheme qscheme, const std::string& name);

void THPQScheme_init(PyObject* module);
