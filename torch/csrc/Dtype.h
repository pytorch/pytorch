#ifndef THP_DTYPE_INC
#define THP_DTYPE_INC

#include <Python.h>
#include "ATen/ATen.h"

struct THPDtype {
  PyObject_HEAD
  at::Type *cdata;
  const char* name;
};

extern PyObject *THPDtypeClass;

#define THPDtype_Check(obj) ((PyObject*)Py_TYPE(obj) == THPDtypeClass)

PyObject * THPDtype_New(at::Type* cdata, const std::string& name);

#ifdef _THP_CORE
bool THPDtype_init(PyObject *module);
#endif

#endif
