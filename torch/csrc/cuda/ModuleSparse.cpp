#include "THCP.h"

static bool THCSPModule_loadClasses(PyObject *module_dict)
{
#define ASSERT_NOT_NULL(ptr) if (!(ptr)) { THPUtils_setError("couldn't load classes"); return false; }
  ASSERT_NOT_NULL(THCSPDoubleTensorClass  = PyMapping_GetItemString(module_dict, (char*)"DoubleTensor"));
  ASSERT_NOT_NULL(THCSPHalfTensorClass    = PyMapping_GetItemString(module_dict, (char*)"HalfTensor"));
  ASSERT_NOT_NULL(THCSPFloatTensorClass   = PyMapping_GetItemString(module_dict, (char*)"FloatTensor"));
  ASSERT_NOT_NULL(THCSPLongTensorClass    = PyMapping_GetItemString(module_dict, (char*)"LongTensor"));
  ASSERT_NOT_NULL(THCSPIntTensorClass     = PyMapping_GetItemString(module_dict, (char*)"IntTensor"));
  ASSERT_NOT_NULL(THCSPShortTensorClass   = PyMapping_GetItemString(module_dict, (char*)"ShortTensor"));
  ASSERT_NOT_NULL(THCSPCharTensorClass    = PyMapping_GetItemString(module_dict, (char*)"CharTensor"));
  ASSERT_NOT_NULL(THCSPByteTensorClass    = PyMapping_GetItemString(module_dict, (char*)"ByteTensor"));

  return true;
#undef ASSERT_NOT_NULL
}

static bool THCSPModule_assignStateless()
{
#define INIT_STATELESS(type)                                                   \
  stateless = PyObject_Call((PyObject*)&TH_CONCAT_3(CudaSparse, type, TensorStatelessType), arg, NULL); \
  if (!stateless) {                                                            \
    THPUtils_setError("stateless method initialization error");                \
    return false;                                                              \
  }                                                                            \
  if (PyObject_SetAttrString(TH_CONCAT_3(THCSP,type,TensorClass), THP_STATELESS_ATTRIBUTE_NAME, stateless) == -1) { \
    THPUtils_setError("stateless method initialization error (on assignment)");\
  }
  PyObject *arg = PyTuple_New(0);
  PyObject *stateless;
  INIT_STATELESS(Double);
  INIT_STATELESS(Float);
  INIT_STATELESS(Half);
  INIT_STATELESS(Long);
  INIT_STATELESS(Int);
  INIT_STATELESS(Short);
  INIT_STATELESS(Char);
  INIT_STATELESS(Byte);
  Py_DECREF(arg);
  return true;
#undef INIT_STATELESS
}

////////////////////////////////////////////////////////////////////////////////
// Sparse Cuda module initialization
////////////////////////////////////////////////////////////////////////////////

bool THCSPModule_initCudaSparse(PyObject *module_dict) {
#define ASSERT_TRUE(cond) if (!(cond)) { return false; }
  ASSERT_TRUE(THCSPModule_loadClasses(module_dict));
  ASSERT_TRUE(THCSPModule_assignStateless());
  return true;
#undef ASSERT_TRUE
}

PyObject * THCSPModule_initExtension(PyObject *self)
{
  PyObject *torch_module = PyImport_ImportModule("torch.cuda.sparse");
  if (!torch_module) {
    THPUtils_setError("class loader couldn't access torch.cuda.sparse module");
    return NULL;
  }
  PyObject* module_dict = PyModule_GetDict(torch_module);
  return PyBool_FromLong(THCSPModule_initCudaSparse(module_dict));
}
