#pragma once

#include <ATen/core/Generator.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/python_headers.h>

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPGenerator {
  PyObject_HEAD at::Generator cdata;
};

// Creates a new Python object wrapping the default at::Generator. The reference
// is borrowed. The caller should ensure that the at::Generator object lifetime
// last at least as long as the Python wrapper.
TORCH_PYTHON_API PyObject* THPGenerator_initDefaultGenerator(
    at::Generator cdata);

#define THPGenerator_Check(obj) PyObject_IsInstance(obj, THPGeneratorClass)

TORCH_PYTHON_API extern PyObject* THPGeneratorClass;

bool THPGenerator_init(PyObject* module);

TORCH_PYTHON_API PyObject* THPGenerator_Wrap(at::Generator gen);

// Creates a new Python object for a Generator. The Generator must not already
// have a PyObject* associated with it.
PyObject* THPGenerator_NewWithVar(PyTypeObject* type, at::Generator gen);
