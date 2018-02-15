#ifndef THP_DTYPE_INC
#define THP_DTYPE_INC

#include <Python.h>
#include "ATen/ATen.h"

struct THPDtype {
  PyObject_HEAD
  at::Type *cdata;
  const char* name;
  bool is_cuda;
  bool is_sparse;
};

extern PyObject *THPDtypeClass;

#define THPDtype_Check(obj) ((PyObject*)Py_TYPE(obj) == THPDtypeClass)

PyObject * THPDtype_New(at::Type* cdata, const std::string& name, bool is_cuda, bool is_sparse);

#ifdef _THP_CORE
bool THPDtype_init(PyObject *module);
#endif

#endif
