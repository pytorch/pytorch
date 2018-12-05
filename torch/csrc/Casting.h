#pragma once

#include "torch/csrc/python_headers.h"

#include <c10/core/Casting.h>

struct THPCasting {
  PyObject_HEAD
  c10::Casting casting;
};

extern PyTypeObject THPCastingType;

inline bool THPCasting_Check(PyObject *obj) {
  return Py_TYPE(obj) == &THPCastingType;
}

PyObject * THPCasting_New(c10::Casting casting);

void THPCasting_init(PyObject *module);
