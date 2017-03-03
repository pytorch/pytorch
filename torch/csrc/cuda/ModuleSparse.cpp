#include "THCP.h"

static bool THCSPModule_loadClasses(PyObject *sparse_module)
{
  if (!THCSPDoubleTensor_postInit(sparse_module)) return false;
  if (!THCSPFloatTensor_postInit(sparse_module)) return false;
#ifdef CUDA_HALF_TENSOR
  if (!THCSPHalfTensor_postInit(sparse_module)) return false;
#endif
  if (!THCSPLongTensor_postInit(sparse_module)) return false;
  if (!THCSPIntTensor_postInit(sparse_module)) return false;
  if (!THCSPShortTensor_postInit(sparse_module)) return false;
  if (!THCSPCharTensor_postInit(sparse_module)) return false;
  if (!THCSPByteTensor_postInit(sparse_module)) return false;
  return true;
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
#ifdef CUDA_HALF_TENSOR
  INIT_STATELESS(Half);
#endif
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

bool THCSPModule_initCudaSparse(PyObject *module) {
#define ASSERT_TRUE(cond) if (!(cond)) { return false; }
  ASSERT_TRUE(THCSPModule_loadClasses(module));
  ASSERT_TRUE(THCSPModule_assignStateless());
  return true;
#undef ASSERT_TRUE
}

PyObject * THCSPModule_initExtension(PyObject *self)
{
  PyObject *module = PyImport_ImportModule("torch.cuda.sparse");
  if (!module) {
    THPUtils_setError("class loader couldn't access torch.cuda.sparse module");
    return NULL;
  }
  return PyBool_FromLong(THCSPModule_initCudaSparse(module));
}
