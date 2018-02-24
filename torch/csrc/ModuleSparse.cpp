#include "THP.h"


////////////////////////////////////////////////////////////////////////////////
// Sparse Stateless Functions
////////////////////////////////////////////////////////////////////////////////

#define IMPLEMENT_SPARSE_STATELESS(name)                                              \
static PyObject * TH_CONCAT_2(THSPModule_, name)(PyObject *_unused, PyObject *args, PyObject *kwargs) \
{                                                                              \
  PyObject *tensor = THSPFloatTensorClass;                                     \
  PyObject *key, *value;                                                       \
  Py_ssize_t pos = 0;                                                          \
  for (int i = 0; i < PyTuple_Size(args); i++) {                               \
    PyObject *item = PyTuple_GET_ITEM(args, i);                                \
    if (THPModule_isTensor(item) || THPVariable_Check(item)) {                 \
      tensor = item;                                                           \
      goto dispatch;                                                           \
    }                                                                          \
  }                                                                            \
  if (kwargs) {                                                                \
    while (PyDict_Next(kwargs, &pos, &key, &value)) {                          \
      if (THPModule_isTensor(value) || THPVariable_Check(value)) {             \
        tensor = value;                                                        \
        goto dispatch;                                                         \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
dispatch:                                                                      \
  return THPUtils_dispatchStateless(tensor, #name, args, kwargs);              \
}

IMPLEMENT_SPARSE_STATELESS(spmm);
IMPLEMENT_SPARSE_STATELESS(sspmm);
IMPLEMENT_SPARSE_STATELESS(sspaddmm);
IMPLEMENT_SPARSE_STATELESS(hspmm);

#undef IMPLEMENT_SPARSE_STATELESS
