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
bool THPStorage_(init)(PyObject *module);
PyObject * THPStorage_(newWeakObject)(THStorage *storage);
#endif

#endif
