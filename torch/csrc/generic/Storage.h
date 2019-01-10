#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Storage.h"
#else

struct THPStorage {
  PyObject_HEAD
  THStorage *cdata;
};

THP_API PyObject * THPStorage_(New)(THStorage *ptr);
extern PyObject *THPStorageClass;

#ifdef _THP_CORE
#include "torch/csrc/Types.h"

bool THPStorage_(init)(PyObject *module);
void THPStorage_(postInit)(PyObject *module);

extern PyTypeObject THPStorageType;
template <> struct THPTypeInfo<THStorage> {
  static PyTypeObject* pyType() { return &THPStorageType; }
  static THStorage* cdata(PyObject* p) { return ((THPStorage*)p)->cdata; }
};
#endif

#endif
