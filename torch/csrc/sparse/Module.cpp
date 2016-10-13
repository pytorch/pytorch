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
  ASSERT_NOT_NULL(THSPDoubleTensorClass  = PyMapping_GetItemString(module_dict, (char*)"DoubleTensor"));
  ASSERT_NOT_NULL(THSPFloatTensorClass   = PyMapping_GetItemString(module_dict, (char*)"FloatTensor"));
  ASSERT_NOT_NULL(THSPLongTensorClass    = PyMapping_GetItemString(module_dict, (char*)"LongTensor"));
  ASSERT_NOT_NULL(THSPIntTensorClass     = PyMapping_GetItemString(module_dict, (char*)"IntTensor"));
  ASSERT_NOT_NULL(THSPShortTensorClass   = PyMapping_GetItemString(module_dict, (char*)"ShortTensor"));
  ASSERT_NOT_NULL(THSPCharTensorClass    = PyMapping_GetItemString(module_dict, (char*)"CharTensor"));
  ASSERT_NOT_NULL(THSPByteTensorClass    = PyMapping_GetItemString(module_dict, (char*)"ByteTensor"));

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

#define ASSERT_TRUE(cond) if (!(cond)) { return false; }
bool THSPModule_initSparse(PyObject *module) {
  ASSERT_TRUE(THSPDoubleTensor_init(module));
  ASSERT_TRUE(THSPFloatTensor_init(module));
  ASSERT_TRUE(THSPLongTensor_init(module));
  ASSERT_TRUE(THSPIntTensor_init(module));
  ASSERT_TRUE(THSPShortTensor_init(module));
  ASSERT_TRUE(THSPCharTensor_init(module));
  ASSERT_TRUE(THSPByteTensor_init(module));
  return true;
}

// Callback for python part. Used for additional initialization of python classes
PyObject *THSPModule_initExtension(PyObject *self)
{
  PyObject *module = PyImport_ImportModule("torch.sparse");
  if (!module) {
    THPUtils_setError("class loader couldn't access torch.sparse module");
    return NULL;
  }

  PyObject* module_dict = PyModule_GetDict(module);
  ASSERT_TRUE(THSPModule_loadClasses(module_dict));
  ASSERT_TRUE(THSPModule_assignStateless());
  Py_RETURN_TRUE;
}
#undef ASSERT_TRUE
