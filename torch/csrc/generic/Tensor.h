#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.h"
#else

#if defined(TH_REAL_IS_HALF) || defined(THD_GENERIC_FILE)
#define GENERATE_SPARSE 0
#else
#define GENERATE_SPARSE 1
#endif

struct THPTensor {
  PyObject_HEAD
  // Invariant: After __new__ (not __init__), this field is always non-NULL.
  THTensor *cdata;
};

#if GENERATE_SPARSE
struct THSPTensor {
  PyObject_HEAD
  // Invariant: After __new__ (not __init__), this field is always non-NULL.
  THSTensor *cdata;
};
#endif

/**
 * Creates a new Python (Sparse) Tensor object using the give THTensor. The
 * returned PyObject* pointer can be safely casted to a THPTensor*.  Note: This
 * "steals" the THTensor* `ptr`.  On error, NULL is returned and the `ptr` ref
 * count is decremented.
 */
THP_API PyObject * THPTensor_(New)(THTensor *ptr);
#if GENERATE_SPARSE
THP_API PyObject * THSPTensor_(New)(THSTensor *ptr);
#endif

/**
 * Creates a new empty Python Tensor object
 */
THP_API PyObject * THPTensor_(NewEmpty)(void);
#if GENERATE_SPARSE
THP_API PyObject * THSPTensor_(NewEmpty)(void);
#endif

THP_API PyObject *THPTensorClass;
#if GENERATE_SPARSE
THP_API PyObject *THSPTensorClass;
#endif

#ifdef _THP_CORE
#include "torch/csrc/Types.h"

// TODO: init stateless in THPTensor_(init) and remove this
THP_API PyTypeObject THPTensorStatelessType;
#if GENERATE_SPARSE
THP_API PyTypeObject THSPTensorStatelessType;
#endif

bool THPTensor_(init)(PyObject *module);
bool THPTensor_(postInit)(PyObject *module);
#if GENERATE_SPARSE
bool THSPTensor_(init)(PyObject *module);
bool THSPTensor_(postInit)(PyObject *module);
#endif

THP_API PyTypeObject THPTensorType;
template <> struct THPTypeInfo<THTensor> {
  static PyTypeObject* pyType() { return &THPTensorType; }
  static THTensor* cdata(PyObject* p) { return ((THPTensor*)p)->cdata; }
};
#endif

#undef GENERATE_SPARSE

#endif
