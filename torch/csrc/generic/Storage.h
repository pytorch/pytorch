#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "torch/csrc/generic/Storage.h"
#else

#include <torch/csrc/StorageDefs.h>

TORCH_PYTHON_API PyObject * THPStorage_(New)(c10::intrusive_ptr<c10::StorageImpl> ptr);
extern PyObject *THPStorageClass;

#include <torch/csrc/Types.h>

bool THPStorage_(init)(PyObject *module);
void THPStorage_(postInit)(PyObject *module);

extern PyTypeObject THPStorageType;

#endif
