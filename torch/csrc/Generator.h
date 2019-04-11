#pragma once

#include <torch/csrc/python_headers.h>
#include <ATen/ATen.h>

#include <torch/csrc/THP_export.h>

struct THPGenerator {
  PyObject_HEAD
  at::Generator *cdata;
  bool owner;  // if true, frees cdata in destructor
};

// Creates a new Python object wrapping the default at::Generator. The reference is
// borrowed. The caller should ensure that the THGenerator* object lifetime
// last at least as long as the Python wrapper.
THP_API PyObject * THPGenerator_initDefaultGenerator(at::Generator* cdata);

#define THPGenerator_Check(obj) \
  PyObject_IsInstance(obj, THPGeneratorClass)

THP_API PyObject *THPGeneratorClass;

#ifdef _THP_CORE
bool THPGenerator_init(PyObject *module);
#endif
