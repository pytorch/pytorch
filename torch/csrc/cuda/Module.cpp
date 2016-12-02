#include <Python.h>

#include <stdbool.h>
#include <unordered_map>
#include <TH/TH.h>
#include <THC/THCCachingAllocator.h>

#include "THCP.h"

THCState *state;

////////////////////////////////////////////////////////////////////////////////
// Class pointer cache
////////////////////////////////////////////////////////////////////////////////

static bool THCPModule_loadClasses(PyObject *module_dict)
{
#define ASSERT_NOT_NULL(ptr) if (!(ptr)) { THPUtils_setError("couldn't load classes"); return false; }
  ASSERT_NOT_NULL(THCPDoubleStorageClass = PyMapping_GetItemString(module_dict, (char*)"DoubleStorage"));
  ASSERT_NOT_NULL(THCPFloatStorageClass  = PyMapping_GetItemString(module_dict, (char*)"FloatStorage"));
  ASSERT_NOT_NULL(THCPHalfStorageClass   = PyMapping_GetItemString(module_dict, (char*)"HalfStorage"));
  ASSERT_NOT_NULL(THCPLongStorageClass   = PyMapping_GetItemString(module_dict, (char*)"LongStorage"));
  ASSERT_NOT_NULL(THCPIntStorageClass    = PyMapping_GetItemString(module_dict, (char*)"IntStorage"));
  ASSERT_NOT_NULL(THCPShortStorageClass  = PyMapping_GetItemString(module_dict, (char*)"ShortStorage"));
  ASSERT_NOT_NULL(THCPCharStorageClass   = PyMapping_GetItemString(module_dict, (char*)"CharStorage"));
  ASSERT_NOT_NULL(THCPByteStorageClass   = PyMapping_GetItemString(module_dict, (char*)"ByteStorage"));

  ASSERT_NOT_NULL(THCPDoubleTensorClass  = PyMapping_GetItemString(module_dict, (char*)"DoubleTensor"));
  ASSERT_NOT_NULL(THCPHalfTensorClass    = PyMapping_GetItemString(module_dict, (char*)"HalfTensor"));
  ASSERT_NOT_NULL(THCPFloatTensorClass   = PyMapping_GetItemString(module_dict, (char*)"FloatTensor"));
  ASSERT_NOT_NULL(THCPLongTensorClass    = PyMapping_GetItemString(module_dict, (char*)"LongTensor"));
  ASSERT_NOT_NULL(THCPIntTensorClass     = PyMapping_GetItemString(module_dict, (char*)"IntTensor"));
  ASSERT_NOT_NULL(THCPShortTensorClass   = PyMapping_GetItemString(module_dict, (char*)"ShortTensor"));
  ASSERT_NOT_NULL(THCPCharTensorClass    = PyMapping_GetItemString(module_dict, (char*)"CharTensor"));
  ASSERT_NOT_NULL(THCPByteTensorClass    = PyMapping_GetItemString(module_dict, (char*)"ByteTensor"));

  return true;
#undef ASSERT_NOT_NULL
}

////////////////////////////////////////////////////////////////////////////////
// Tensor stateless methods
////////////////////////////////////////////////////////////////////////////////

static bool THCPModule_assignStateless()
{
#define INIT_STATELESS(type) INIT_STATELESS_DETAIL(type, TH_CONCAT_2(Cuda, type))
#define INIT_STATELESS_DETAIL(type,ctype)                                      \
  stateless = PyObject_Call((PyObject*)&TH_CONCAT_2(ctype, TensorStatelessType), arg, NULL); \
  if (!stateless) {                                                            \
    THPUtils_setError("stateless method initialization error");                \
    return false;                                                              \
  }                                                                            \
  if (PyObject_SetAttrString(TH_CONCAT_3(THCP,type,TensorClass), THP_STATELESS_ATTRIBUTE_NAME, stateless) == -1) { \
    THPUtils_setError("stateless method initialization error (on assignment)");\
  }
  PyObject *arg = PyTuple_New(0);
  PyObject *stateless;
  INIT_STATELESS(Double);
  INIT_STATELESS_DETAIL(Float, Cuda);
  INIT_STATELESS(Long);
  INIT_STATELESS(Int);
  INIT_STATELESS(Short);
  INIT_STATELESS(Char);
  INIT_STATELESS(Byte);
  Py_DECREF(arg);
  return true;
#undef INIT_STATELESS_DETAIL
#undef INIT_STATELESS
}

////////////////////////////////////////////////////////////////////////////////
// Additional copy handlers
////////////////////////////////////////////////////////////////////////////////

#include "ModuleCopy.cpp"

////////////////////////////////////////////////////////////////////////////////
// CUDA management methods
////////////////////////////////////////////////////////////////////////////////

void THCPModule_setDevice(int device)
{
  THCudaCheck(cudaSetDevice(device));
}

PyObject * THCPModule_setDevice_wrap(PyObject *self, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to setDevice");
  long device = THPUtils_unpackLong(arg);

  THCPModule_setDevice(device);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_getDevice_wrap(PyObject *self)
{
  HANDLE_TH_ERRORS
  int device;
  THCudaCheck(cudaGetDevice(&device));
  return PyLong_FromLong(device);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_getDeviceCount_wrap(PyObject *self)
{
  HANDLE_TH_ERRORS
  int ndevice;
  THCudaCheck(cudaGetDeviceCount(&ndevice));
  return PyLong_FromLong(ndevice);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_getCurrentStream_wrap(PyObject *self)
{
  HANDLE_TH_ERRORS
  THCStream* stream = THCState_getStream(state);
  return PyLong_FromVoidPtr(stream);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_setStream_wrap(PyObject *self, PyObject *obj)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(PyLong_Check(obj), "invalid stream");
  THCStream* stream = (THCStream *)PyLong_AsVoidPtr(obj);
  THCState_setStream(state, stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_isDriverSufficient(PyObject *self)
{
  int count;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err == cudaErrorInsufficientDriver) {
    return PyBool_FromLong(0);
  }
  return PyBool_FromLong(1);
}

PyObject * THCPModule_getDriverVersion(PyObject *self)
{
  int driverVersion = -1;
  cudaError_t err = cudaDriverGetVersion(&driverVersion);
  if (err != cudaSuccess) {
    PyErr_Format(PyExc_RuntimeError,
                    "Error calling cudaDriverGetVersion: %d %s",
                    err, cudaGetErrorString(err));
    return NULL;
  }
  return PyLong_FromLong((long) driverVersion);
}

PyObject * THCPModule_getRNGState(PyObject *_unused)
{
  HANDLE_TH_ERRORS
  THPByteTensorPtr res = (THPByteTensor *)THPByteTensor_NewEmpty();
  if (!res) return NULL;
  THCRandom_getRNGState(state, res->cdata);
  return (PyObject *)res.release();
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_setRNGState(PyObject *_unused, PyObject *_new_rng_state)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPByteTensor_Check(_new_rng_state), "set_rng_state expects a "
          "torch.ByteTensor, but got %s", THPUtils_typename(_new_rng_state));
  THByteTensor *new_rng_state = ((THPByteTensor*)_new_rng_state)->cdata;
  THCRandom_setRNGState(state, new_rng_state);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_manualSeed(PyObject *_unused, PyObject *seed)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(seed), "manual_seed expected a long, "
          "but got %s", THPUtils_typename(seed));
  THCRandom_manualSeed(state, THPUtils_unpackLong(seed));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_manualSeedAll(PyObject *_unused, PyObject *seed)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(seed), "manual_seed expected a long, "
          "but got %s", THPUtils_typename(seed));
  THCRandom_manualSeedAll(state, THPUtils_unpackLong(seed));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_seed(PyObject *_unused)
{
  HANDLE_TH_ERRORS
  return PyLong_FromUnsignedLong(THCRandom_seed(state));
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_seedAll(PyObject *_unused)
{
  HANDLE_TH_ERRORS
  return PyLong_FromUnsignedLong(THCRandom_seedAll(state));
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_initialSeed(PyObject *_unused)
{
  HANDLE_TH_ERRORS
  return PyLong_FromUnsignedLong(THCRandom_initialSeed(state));
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_cudaHostAllocator(PyObject *_unused)
{
  HANDLE_TH_ERRORS
  THAllocator* allocator = THCState_getCudaHostAllocator(state);
  return PyLong_FromVoidPtr(allocator);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_cudaSynchronize(PyObject *_unused)
{
  HANDLE_TH_ERRORS
  THCudaCheck(cudaDeviceSynchronize());
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_cudaSleep(PyObject *_unused, PyObject *cycles)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(cycles), "torch.cuda._sleep(): expected 'int'");
  THC_sleep(LIBRARY_STATE THPUtils_unpackLong(cycles));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_getLibPath(PyObject *_unused)
{
#define _STR(x) #x
#define STR(x) _STR(x)
#if PY_MAJOR_VERSION == 2
  return PyString_FromString(STR(CUDA_LIB_PATH));
#else
  return PyUnicode_FromString(STR(CUDA_LIB_PATH));
#endif
#undef STR
#undef _STR
}

////////////////////////////////////////////////////////////////////////////////
// Cuda module initialization
////////////////////////////////////////////////////////////////////////////////

bool THCPModule_initCuda(PyObject *module_dict) {
#define ASSERT_TRUE(cond) if (!(cond)) { return false; }
  state = THCState_alloc();
  THCState_setDeviceAllocator(state, THCCachingAllocator_get());
  state->cudaHostAllocator = &THCCachingHostAllocator;
  THCudaInit(state);

#ifdef USE_MAGMA
  THCMagma_init(state);
  ASSERT_TRUE(PyDict_SetItemString(module_dict, "hasMagma", PyBool_FromLong(true)) != -1);
#else
  ASSERT_TRUE(PyDict_SetItemString(module_dict, "hasMagma", PyBool_FromLong(false)) != -1);
#endif

#ifdef CUDA_HALF_TENSOR
  ASSERT_TRUE(PyDict_SetItemString(module_dict, "hasHalf", PyBool_FromLong(true)) != -1);
#else
  ASSERT_TRUE(PyDict_SetItemString(module_dict, "hasHalf", PyBool_FromLong(false)) != -1);
#endif

  ASSERT_TRUE(THCPModule_loadClasses(module_dict));
  ASSERT_TRUE(THCPModule_assignStateless());
  ASSERT_TRUE(THCPModule_initCopy());

  ASSERT_TRUE(PyDict_SetItemString(module_dict, "_state_cdata", PyLong_FromVoidPtr(state)) != -1);

  // TODO: register THCudaShutdown handler at exit
  return true;
#undef ASSERT_TRUE
}

// Callback for python part. Used for additional initialization of python classes
PyObject * THCPModule_initExtension(PyObject *self)
{
  PyObject *torch_module = PyImport_ImportModule("torch.cuda");
  if (!torch_module) {
    THPUtils_setError("class loader couldn't access torch module");
    return NULL;
  }
  PyObject* module_dict = PyModule_GetDict(torch_module);
  return PyBool_FromLong(THCPModule_initCuda(module_dict));
}
