#include <Python.h>

#include <stdbool.h>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <TH/TH.h>
#include <ATen/ATen.h>
#include <THC/THCCachingAllocator.h>
#ifdef WITH_NCCL
#include <nccl.h>
#endif

#include "THCP.h"

#include "ModuleSparse.cpp"

THCState *state;

////////////////////////////////////////////////////////////////////////////////
// Class pointer cache
////////////////////////////////////////////////////////////////////////////////

static bool THCPModule_loadClasses(PyObject *torch_module)
{
#define ASSERT_NOT_NULL(ptr) if (!(ptr)) { THPUtils_setError("couldn't load classes"); return false; }
  ASSERT_NOT_NULL(THCPDoubleStorageClass = PyObject_GetAttrString(torch_module, (char*)"DoubleStorage"));
  ASSERT_NOT_NULL(THCPFloatStorageClass  = PyObject_GetAttrString(torch_module, (char*)"FloatStorage"));
  ASSERT_NOT_NULL(THCPHalfStorageClass   = PyObject_GetAttrString(torch_module, (char*)"HalfStorage"));
  ASSERT_NOT_NULL(THCPLongStorageClass   = PyObject_GetAttrString(torch_module, (char*)"LongStorage"));
  ASSERT_NOT_NULL(THCPIntStorageClass    = PyObject_GetAttrString(torch_module, (char*)"IntStorage"));
  ASSERT_NOT_NULL(THCPShortStorageClass  = PyObject_GetAttrString(torch_module, (char*)"ShortStorage"));
  ASSERT_NOT_NULL(THCPCharStorageClass   = PyObject_GetAttrString(torch_module, (char*)"CharStorage"));
  ASSERT_NOT_NULL(THCPByteStorageClass   = PyObject_GetAttrString(torch_module, (char*)"ByteStorage"));

  if (!THCPDoubleTensor_postInit(torch_module)) return false;
  if (!THCPFloatTensor_postInit(torch_module)) return false;
  if (!THCPHalfTensor_postInit(torch_module)) return false;
  if (!THCPLongTensor_postInit(torch_module)) return false;
  if (!THCPIntTensor_postInit(torch_module)) return false;
  if (!THCPShortTensor_postInit(torch_module)) return false;
  if (!THCPCharTensor_postInit(torch_module)) return false;
  if (!THCPByteTensor_postInit(torch_module)) return false;

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
  INIT_STATELESS(Half);
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
  if (cudaGetDeviceCount(&ndevice) != cudaSuccess) {
    cudaGetLastError();
    ndevice = 0;
  }
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
  THPByteTensorPtr res((THPByteTensor *)THPByteTensor_NewEmpty());
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

// We need to ensure that as long as a thread will NEVER loose the GIL as long as
// it holds the CUDA mutex. Otherwise another thread might be scheduled and try to
// e.g. allocate a new tensor which will cause a deadlock. It's enough to have a
// single global, because it can be only set once (cudaMutex is not recursive)
// by the thread that owns the mutex (obviously there can be only one such thread).
static PyGILState_STATE cudaMutexGILState;

PyObject * THCPModule_cudaLockMutex(PyObject *module)
{
  auto mutex = THCCachingAllocator_getCudaFreeMutex();
  // This has to be a busy loop because we **absolutely need to** hold the GIL
  // or it's a recipe for a deadlock otherwise (if we let other Python threads
  // run while we have the cudaMutex, but not the GIL, they might try to e.g.
  // free a CUDA tensor and acquire the cudaMutex without giving up the GIL,
  // because it happens deep within THC).
  while (true) {
    if (mutex->try_lock())
      break;
    {
      AutoNoGIL no_gil;
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  cudaMutexGILState = PyGILState_Ensure();
  Py_RETURN_NONE;
}

PyObject * THCPModule_cudaUnlockMutex(PyObject *module)
{
  auto mutex = THCCachingAllocator_getCudaFreeMutex();
  PyGILState_Release(cudaMutexGILState);
  mutex->unlock();
  Py_RETURN_NONE;
}

////////////////////////////////////////////////////////////////////////////////
// Cuda module initialization
////////////////////////////////////////////////////////////////////////////////

bool THCPModule_initCuda(PyObject *torch_module) {
  HANDLE_TH_ERRORS
#define ASSERT_TRUE(cond) if (!(cond)) { return false; }
  state = at::globalContext().lazyInitCUDA();

#ifdef USE_MAGMA
  THCMagma_init(state);
  ASSERT_TRUE(PyObject_SetAttrString(torch_module, "has_magma", PyBool_FromLong(true)) != -1);
#else
  ASSERT_TRUE(PyObject_SetAttrString(torch_module, "has_magma", PyBool_FromLong(false)) != -1);
#endif

#ifdef CUDA_HALF_TENSOR
  ASSERT_TRUE(PyObject_SetAttrString(torch_module, "has_half", PyBool_FromLong(true)) != -1);
#else
  ASSERT_TRUE(PyObject_SetAttrString(torch_module, "has_half", PyBool_FromLong(false)) != -1);
#endif

  ASSERT_TRUE(THCPModule_loadClasses(torch_module));
  ASSERT_TRUE(THCPModule_assignStateless());

  ASSERT_TRUE(PyObject_SetAttrString(torch_module, "_state_cdata", PyLong_FromVoidPtr(state)) != -1);

  // TODO: register THCudaShutdown handler at exit
  return true;
#undef ASSERT_TRUE
  END_HANDLE_TH_ERRORS_RET(false)
}

// Callback for python part. Used for additional initialization of python classes
PyObject * THCPModule_initExtension(PyObject *self)
{
  PyObject *torch_module = PyImport_ImportModule("torch.cuda");
  if (!torch_module) {
    THPUtils_setError("class loader couldn't access torch module");
    return NULL;
  }
  if (!THCPModule_initCuda(torch_module)) {
    return NULL;
  }
  Py_RETURN_NONE;
}

#ifdef WITH_NCCL
void THCPModule_useNccl()
{
  // Use NCCL to ensure that the symbols are loaded
  ncclUniqueId uniqueId;
  ncclGetUniqueId(&uniqueId);
}
#endif

PyObject * THCPModule_getCurrentBlasHandle_wrap(PyObject *self)
{
  HANDLE_TH_ERRORS
  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  return PyLong_FromVoidPtr(handle);
  END_HANDLE_TH_ERRORS
}
