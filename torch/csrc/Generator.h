#pragma once

#include <ATen/core/Generator.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/python_headers.h>

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPGenerator {
  PyObject_HEAD
  at::Generator cdata;
  PyObject* weakreflist;
};

// Creates a new Python object wrapping the default at::Generator. The reference
// is borrowed. The caller should ensure that the at::Generator object lifetime
// last at least as long as the Python wrapper.
TORCH_PYTHON_API PyObject* THPGenerator_initDefaultGenerator(
    const at::Generator& cdata);

#define THPGenerator_Check(obj) PyObject_IsInstance(obj, THPGeneratorClass)

TORCH_PYTHON_API extern PyObject* THPGeneratorClass;

bool THPGenerator_init(PyObject* module);

TORCH_PYTHON_API PyObject* THPGenerator_Wrap(const at::Generator& gen);

TORCH_PYTHON_API at::Generator THPGenerator_Unwrap(PyObject* state);

// Returns a new reference to a Python wrapper for gen, reusing a live cached
// wrapper if one is found. Otherwise creates and caches a new wrapper with the
// given type.
// See Note [Generator PyObject cache] in Generator.cpp.
PyObject* THPGenerator_NewWithVar(PyTypeObject* type, at::Generator gen);
