#ifndef THP_STORAGE_INC
#define THP_STORAGE_INC

#include <torch/csrc/Types.h>

#define THPStorageStr "torch.UntypedStorage"
#define THPStorageBaseStr "StorageBase"

struct THPStorage {
  PyObject_HEAD c10::StorageImpl* cdata;
};

TORCH_PYTHON_API PyObject* THPStorage_New(
    c10::intrusive_ptr<c10::StorageImpl> ptr);
extern PyObject* THPStorageClass;

bool THPStorage_init(PyObject* module);
void THPStorage_postInit(PyObject* module);

extern PyTypeObject THPStorageType;

#endif
