#include <Python.h>

#include <stdbool.h>
#include <unordered_map>
#include <THS/THS.h>

#include "THSP.h"

PyObject* sparse_tensor_classes;

static bool THSPModule_loadClasses(PyObject *module_dict)
{
#define ASSERT_NOT_NULL(ptr) if (!(ptr)) { THPUtils_setError("couldn't load classes"); return false; }
  ASSERT_NOT_NULL(sparse_tensor_classes = PyMapping_GetItemString(module_dict, (char*)"_sparse_tensor_classes"));
  ASSERT_NOT_NULL(THSPDoubleTensorClass  = PyMapping_GetItemString(module_dict, (char*)"SparseDoubleTensor"));
  ASSERT_NOT_NULL(THSPFloatTensorClass   = PyMapping_GetItemString(module_dict, (char*)"SparseFloatTensor"));
  ASSERT_NOT_NULL(THSPLongTensorClass    = PyMapping_GetItemString(module_dict, (char*)"SparseLongTensor"));
  ASSERT_NOT_NULL(THSPIntTensorClass     = PyMapping_GetItemString(module_dict, (char*)"SparseIntTensor"));
  ASSERT_NOT_NULL(THSPShortTensorClass   = PyMapping_GetItemString(module_dict, (char*)"SparseShortTensor"));
  ASSERT_NOT_NULL(THSPCharTensorClass    = PyMapping_GetItemString(module_dict, (char*)"SparseCharTensor"));
  ASSERT_NOT_NULL(THSPByteTensorClass    = PyMapping_GetItemString(module_dict, (char*)"SparseByteTensor"));

  return true;
#undef ASSERT_NOT_NULL
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

bool THSPModule_initSparse(PyObject *module_dict) {
#define ASSERT_TRUE(cond) if (!(cond)) { return false; }
  ASSERT_TRUE(THSPModule_loadClasses(module_dict));
  ASSERT_TRUE(THSPModule_assignStateless());
  return true;
#undef ASSERT_TRUE
}

#include <stdio.h>
// Callback for python part. Used for additional initialization of python classes
bool THSPModule_initExtension(PyObject *self)
{
  PyObject *torch_module = PyImport_ImportModule("torch");
  if (!torch_module) {
    THPUtils_setError("class loader couldn't access torch module");
    return NULL;
  }
  PyObject* module_dict = PyModule_GetDict(torch_module);
  bool res = THSPModule_initSparse(module_dict);
  printf("%i\n", res);
  return res;
}
