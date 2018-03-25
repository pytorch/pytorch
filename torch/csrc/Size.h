#pragma once

#include <Python.h>
#include "stdint.h"

extern PyTypeObject THPSizeType;

#define THPSize_Check(obj) (Py_TYPE(obj) == &THPSizeType)

PyObject * THPSize_New(int dim, const int64_t *sizes);

#ifdef _THP_CORE
void THPSize_init(PyObject *module);
#endif
