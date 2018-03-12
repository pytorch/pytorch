#pragma once

#include <Python.h>
#include <ATen/ATen.h>

#include "THP_export.h"

struct THPGenerator {
  PyObject_HEAD
  at::Generator *cdata;
  bool owner;  // if true, frees cdata in destructor
};

#define THPGenerator_Check(obj) \
  PyObject_IsInstance(obj, THPGeneratorClass)

#define THPGenerator_TH_CData(obj) \
  (THGenerator*)((THPGenerator*)obj)->cdata->unsafeGetTH()

THP_API PyObject * THPGenerator_New();

// Creates a new Python object wrapping the at::Generator. The reference is
// borrowed. The caller should ensure that the THGenerator* object lifetime
// last at least as long as the Python wrapper.
THP_API PyObject * THPGenerator_NewWithGenerator(at::Generator& cdata);

THP_API PyObject *THPGeneratorClass;

#ifdef _THP_CORE
bool THPGenerator_init(PyObject *module);
#endif
