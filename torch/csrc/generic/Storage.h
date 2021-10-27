#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "torch/csrc/generic/Storage.h"
#else

#include <torch/csrc/StorageDefs.h>

THP_API PyObject * THPStorage_(New)(THWStorage *ptr);
extern PyObject *THPStorageClass;

#include <torch/csrc/Types.h>



extern PyTypeObject THPStorageType;

#endif
