#ifndef THP_STORAGE_INC
#define THP_STORAGE_INC

#include <Python.h>
#include <c10/core/Storage.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/Types.h>

#define THPStorageStr "torch.UntypedStorage"

struct THPStorage {
  PyObject_HEAD
  c10::MaybeOwned<c10::Storage> cdata;
  bool is_hermetic;
};

TORCH_PYTHON_API PyObject* THPStorage_Wrap(c10::Storage storage);
TORCH_PYTHON_API PyObject* THPStorage_NewWithStorage(
    PyTypeObject* type,
    c10::Storage _storage,
    bool allow_preexisting_pyobj = false);
TORCH_PYTHON_API extern PyTypeObject* THPStorageClass;

inline bool THPStorage_CheckTypeExact(PyTypeObject* tp) {
  return tp == THPStorageClass;
}

inline bool THPStorage_CheckExact(PyObject* obj) {
  return THPStorage_CheckTypeExact(Py_TYPE(obj));
}

inline bool THPStorage_Check(PyObject* obj) {
  if (!THPStorageClass)
    return false;

  const auto result = PyObject_IsInstance(obj, (PyObject*)THPStorageClass);
  if (result == -1)
    throw python_error();
  return result;
}

bool THPStorage_init(PyObject* module);
void THPStorage_postInit(PyObject* module);

void THPStorage_assertNotNull(THPStorage* storage);
TORCH_PYTHON_API void THPStorage_assertNotNull(PyObject* obj);

TORCH_PYTHON_API extern PyTypeObject THPStorageType;

inline const c10::Storage& THPStorage_Unpack(THPStorage* storage) {
  return *storage->cdata;
}

inline const c10::Storage& THPStorage_Unpack(PyObject* obj) {
  return THPStorage_Unpack(reinterpret_cast<THPStorage*>(obj));
}

#endif
