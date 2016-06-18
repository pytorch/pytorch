#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.h"
#else

struct THPTensor {
  PyObject_HEAD
  THTensor *cdata;
};

extern PyTypeObject THPTensorType;
extern PyTypeObject THPTensorStatelessType;
extern PyObject *THPTensorClass;

bool THPTensor_(init)(PyObject *module);
PyObject * THPTensor_(newObject)(THTensor *tensor);
bool THPTensor_(IsSubclass)(PyObject *tensor);

#endif
