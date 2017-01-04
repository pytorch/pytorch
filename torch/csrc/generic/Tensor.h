#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.h"
#else

struct THPTensor {
  PyObject_HEAD
  THTensor *cdata;
};

struct THSPTensor {
  PyObject_HEAD
  THSTensor *cdata;
};

/**
 * Creates a new Python (Sparse) Tensor object using the give THTensor. The
 * returned PyObject* pointer can be safely casted to a THPTensor*.  Note: This
 * "steals" the THTensor* `ptr`.  On error, NULL is returned and the `ptr` ref
 * count is decremented.
 */
THP_API PyObject * THPTensor_(New)(THTensor *ptr);
THP_API PyObject * THSPTensor_(New)(THSTensor *ptr);

/**
 * Creates a new empty Python Tensor object
 */
THP_API PyObject * THPTensor_(NewEmpty)(void);
THP_API PyObject * THSPTensor_(NewEmpty)(void);

extern PyObject *THPTensorClass;
extern PyObject *THSPTensorClass;

#ifdef _THP_CORE
#include "torch/csrc/Types.h"

// TODO: init stateless in THPTensor_(init) and remove this
extern PyTypeObject THPTensorStatelessType;
extern PyTypeObject THSPTensorStatelessType;
bool THPTensor_(init)(PyObject *module);
bool THSPTensor_(init)(PyObject *module);

extern PyTypeObject THPTensorType;
template <> struct THPTypeInfo<THTensor> {
  static PyTypeObject* pyType() { return &THPTensorType; }
  static THTensor* cdata(PyObject* p) { return ((THPTensor*)p)->cdata; }
};
#endif

#endif
