#include "THP.h"

PyObject* sparse_tensor_classes;

////////////////////////////////////////////////////////////////////////////////
// SPARSE MODULE INITIALIZATION
////////////////////////////////////////////////////////////////////////////////

static bool THSPModule_loadClasses(PyObject *sparse_module)
{
  if (!THSPDoubleTensor_postInit(sparse_module)) return false;
  if (!THSPFloatTensor_postInit(sparse_module)) return false;
  if (!THSPLongTensor_postInit(sparse_module)) return false;
  if (!THSPIntTensor_postInit(sparse_module)) return false;
  if (!THSPShortTensor_postInit(sparse_module)) return false;
  if (!THSPCharTensor_postInit(sparse_module)) return false;
  if (!THSPByteTensor_postInit(sparse_module)) return false;
  return true;
}

static bool THSPModule_assignStateless()
{
#define INIT_STATELESS(type)                                                   \
  stateless = PyObject_Call((PyObject*)&TH_CONCAT_3(Sparse, type, TensorStatelessType), arg, NULL); \
  if (!stateless) {                                                            \
    THPUtils_setError("stateless method initialization error");                \
    return false;                                                              \
  }                                                                            \
  if (PyObject_SetAttrString(TH_CONCAT_3(THSP,type,TensorClass), THP_STATELESS_ATTRIBUTE_NAME, stateless) == -1) { \
    THPUtils_setError("stateless method initialization error (on assignment)");\
  }
  PyObject *arg = PyTuple_New(0);
  PyObject *stateless;
  INIT_STATELESS(Double);
  INIT_STATELESS(Float);
  INIT_STATELESS(Long);
  INIT_STATELESS(Int);
  INIT_STATELESS(Short);
  INIT_STATELESS(Char);
  INIT_STATELESS(Byte);
  Py_DECREF(arg);
  return true;
#undef INIT_STATELESS
}

// Callback for python part. Used for additional initialization of python classes
PyObject *THSPModule_initExtension(PyObject *self)
{
  PyObject *module = PyImport_ImportModule("torch.sparse");
  if (!module) return NULL;
  if (!THSPModule_loadClasses(module)) return NULL;
  if (!THSPModule_assignStateless()) return NULL;
  Py_RETURN_NONE;
}

////////////////////////////////////////////////////////////////////////////////
// Sparse Stateless Functions
////////////////////////////////////////////////////////////////////////////////

bool THPModule_isSparseTensor(PyObject *obj)
{
  int result = PySet_Contains(sparse_tensor_classes, (PyObject*)Py_TYPE(obj));
  if (result == -1)
    throw std::logic_error("FATAL: sparse_tensor_classes isn't a set!");
  return result;
}


#define IMPLEMENT_SPARSE_STATELESS(name)                                              \
static PyObject * TH_CONCAT_2(THSPModule_, name)(PyObject *_unused, PyObject *args, PyObject *kwargs) \
{                                                                              \
  PyObject *tensor = THSPFloatTensorClass;                                     \
  PyObject *key, *value;                                                       \
  Py_ssize_t pos = 0;                                                          \
  for (int i = 0; i < PyTuple_Size(args); i++) {                               \
    PyObject *item = PyTuple_GET_ITEM(args, i);                                \
    if (THPModule_isTensor(item) || THPVariable_Check(item)) {                 \
      tensor = item;                                                           \
      goto dispatch;                                                           \
    }                                                                          \
  }                                                                            \
  if (kwargs) {                                                                \
    while (PyDict_Next(kwargs, &pos, &key, &value)) {                          \
      if (THPModule_isTensor(value) || THPVariable_Check(value)) {             \
        tensor = value;                                                        \
        goto dispatch;                                                         \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
dispatch:                                                                      \
  return THPUtils_dispatchStateless(tensor, #name, args, kwargs);              \
}

IMPLEMENT_SPARSE_STATELESS(spmm);
IMPLEMENT_SPARSE_STATELESS(sspmm);
IMPLEMENT_SPARSE_STATELESS(sspaddmm);

#undef IMPLEMENT_SPARSE_STATELESS
