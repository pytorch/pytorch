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

////////////////////////////////////////////////////////////////////////////////
// Sparse Cuda module initialization
////////////////////////////////////////////////////////////////////////////////

bool THCSPModule_initCudaSparse(PyObject *module) {
#define ASSERT_TRUE(cond) if (!(cond)) { return false; }
  ASSERT_TRUE(THCSPModule_loadClasses(module));
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
  if (!THCSPModule_initCudaSparse(module)) {
    return NULL;
  }
  Py_RETURN_NONE;
}
