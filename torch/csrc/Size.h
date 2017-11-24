#ifndef THP_SIZE_INC
#define THP_SIZE_INC

#include <Python.h>
#include "stdint.h"

extern PyObject *THPSizeClass;

#define THPSize_Check(obj) ((PyObject*)Py_TYPE(obj) == THPSizeClass)

PyObject * THPSize_New(int dim, const int64_t *sizes);

#ifdef _THP_CORE
bool THPSize_init(PyObject *module);
#endif

#endif
