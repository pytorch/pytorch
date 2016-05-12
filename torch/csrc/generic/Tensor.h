#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.h"
#else

typedef struct {
  PyObject_HEAD
  THTensor *cdata;
} THPTensor;

extern PyTypeObject THPTensorType;
extern PyTypeObject THPTensorStatelessType;

bool THPTensor_(init)(PyObject *module);
PyObject * THPTensor_(newObject)(THTensor *tensor);
bool THPTensor_(IsSubclass)(PyObject *tensor);

#endif
