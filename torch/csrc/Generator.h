#pragma once

#include <torch/csrc/python_headers.h>
#include <ATen/ATen.h>

#include <torch/csrc/THP_export.h>

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPGenerator {
  PyObject_HEAD
  at::Generator cdata;
};

// Creates a new Python object wrapping the default at::Generator. The reference is
// borrowed. The caller should ensure that the at::Generator object lifetime
// last at least as long as the Python wrapper.
THP_API PyObject * THPGenerator_initDefaultGenerator(at::Generator cdata);

#define THPGenerator_Check(obj) \
  PyObject_IsInstance(obj, THPGeneratorClass)

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
THP_API PyObject *THPGeneratorClass;

bool THPGenerator_init(PyObject *module);

THP_API PyObject * THPGenerator_Wrap(at::Generator gen);

// Creates a new Python object for a Generator. The Generator must not already
// have a PyObject* associated with it.
PyObject* THPGenerator_NewWithVar(PyTypeObject* type, at::Generator gen);
