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

// Returns a new reference to the Python object of a default at::Generator.
TORCH_PYTHON_API PyObject* THPGenerator_initDefaultGenerator(
    const at::Generator& cdata);

#define THPGenerator_Check(obj) PyObject_IsInstance(obj, THPGeneratorClass)

TORCH_PYTHON_API extern PyObject* THPGeneratorClass;

bool THPGenerator_init(PyObject* module);

TORCH_PYTHON_API PyObject* THPGenerator_Wrap(const at::Generator& gen);

TORCH_PYTHON_API at::Generator THPGenerator_Unwrap(PyObject* state);

// Returns a new reference to the Python object of a Generator, creating one of
// the given type if the Generator does not have one yet.
PyObject* THPGenerator_NewWithVar(PyTypeObject* type, at::Generator gen);
