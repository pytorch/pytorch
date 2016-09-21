#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.h"
#else

struct THPTensor {
  PyObject_HEAD
  THTensor *cdata;
};

THP_API PyObject * THPTensor_(New)(THTensor *ptr);
extern PyObject *THPTensorClass;

#ifdef _THP_CORE
// TODO: init stateless in THPTensor_(init) and remove this
extern PyTypeObject THPTensorStatelessType;
bool THPTensor_(init)(PyObject *module);
#endif

#endif
