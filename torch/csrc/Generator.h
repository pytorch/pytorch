#pragma once

#include "torch/csrc/python_headers.h"
#include <ATen/ATen.h>

#include "THP_export.h"

struct THPGenerator {
  PyObject_HEAD
  at::Generator *cdata;
  bool owner;  // if true, frees cdata in destructor
};

#define THPGenerator_Check(obj) \
  PyObject_IsInstance(obj, THPGeneratorClass)

THP_API PyObject *THPGeneratorClass;

#ifdef _THP_CORE
bool THPGenerator_init(PyObject *module);
#endif
