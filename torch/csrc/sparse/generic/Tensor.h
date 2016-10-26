#ifndef THS_GENERIC_FILE
#define THS_GENERIC_FILE "torch/csrc/sparse/generic/Tensor.h"
#else

struct THSPTensor {
  PyObject_HEAD
  THSTensor *cdata;
};

THP_API PyObject * THSPTensor_(New)(THSTensor *ptr);
extern PyObject *THSPTensorClass;

#ifdef _THP_CORE
extern PyTypeObject THSPTensorStatelessType;
bool THSPTensor_(init)(PyObject *module);
//PyObject * THSPTensor_(newWeakObject)(THSTensor *tensor);
#endif

#endif

