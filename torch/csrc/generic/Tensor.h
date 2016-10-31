#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.h"
#else

struct THPTensor {
  PyObject_HEAD
  THTensor *cdata;
};

/**
 * Creates a new Python Tensor object using the give THTensor. The returned
 * PyObject* pointer can be safely casted to a THPTensor*.
 * Note: This "steals" the THTensor* `ptr`.
 * On error, NULL is returned and the `ptr` ref count is decremented.
 */
THP_API PyObject * THPTensor_(New)(THTensor *ptr);

/**
 * Creates a new empty Python Tensor object
 */
THP_API PyObject * THPTensor_(NewEmpty)(void);

extern PyObject *THPTensorClass;

#ifdef _THP_CORE
// TODO: init stateless in THPTensor_(init) and remove this
extern PyTypeObject THPTensorStatelessType;
bool THPTensor_(init)(PyObject *module);
#endif

#endif
