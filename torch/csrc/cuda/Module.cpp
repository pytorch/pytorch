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

#include "torch/csrc/utils/python_strings.h"
#include "ModuleSparse.cpp"

THCState *state;

////////////////////////////////////////////////////////////////////////////////
// Class pointer cache
////////////////////////////////////////////////////////////////////////////////

static bool THCPModule_loadClasses(PyObject *torch_module)
{
#define ASSERT_NOT_NULL(ptr) if (!(ptr)) { THPUtils_setError("couldn't load classes"); return false; }
  if (!THCPDoubleTensor_postInit(torch_module)) return false;
  if (!THCPFloatTensor_postInit(torch_module)) return false;
  if (!THCPHalfTensor_postInit(torch_module)) return false;
  if (!THCPLongTensor_postInit(torch_module)) return false;
  if (!THCPIntTensor_postInit(torch_module)) return false;
  if (!THCPShortTensor_postInit(torch_module)) return false;
  if (!THCPCharTensor_postInit(torch_module)) return false;
  if (!THCPByteTensor_postInit(torch_module)) return false;

  THCPDoubleStorage_postInit(torch_module);
  THCPFloatStorage_postInit(torch_module);
  THCPHalfStorage_postInit(torch_module);
  THCPLongStorage_postInit(torch_module);
  THCPIntStorage_postInit(torch_module);
  THCPShortStorage_postInit(torch_module);
  THCPCharStorage_postInit(torch_module);
  THCPByteStorage_postInit(torch_module);

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
  int64_t device = THPUtils_unpackLong(arg);

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

PyObject * THCPModule_getDeviceName_wrap(PyObject *self, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to getDeviceName");
  long device = THPUtils_unpackLong(arg);

  cudaDeviceProp prop;
  THCudaCheck(cudaGetDeviceProperties(&prop, device));
  return THPUtils_packString(prop.name);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_getDeviceCapability_wrap(PyObject *self, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to getDeviceCapability");
  long device = THPUtils_unpackLong(arg);

  cudaDeviceProp prop;
  THCudaCheck(cudaGetDeviceProperties(&prop, device));
  return Py_BuildValue("(ii)", prop.major, prop.minor);
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
  return PyLong_FromLong((int64_t) driverVersion);
}

PyObject * THCPModule_getCompiledVersion(PyObject *self)
{
  return PyLong_FromLong((long) CUDA_VERSION);
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

PyObject * THCPModule_emptyCache(PyObject *_unused)
{
  HANDLE_TH_ERRORS
  auto device_allocator = THCState_getDeviceAllocator(state);
  THCudaCheck(device_allocator->emptyCache(device_allocator->state));
  END_HANDLE_TH_ERRORS
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
#include "nccl.h"

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

static struct PyMethodDef _THCPModule_methods[] = {
  {"_cuda_init",        (PyCFunction)THCPModule_initExtension,    METH_NOARGS,  NULL},
  {"_cuda_setDevice",   (PyCFunction)THCPModule_setDevice_wrap,   METH_O,       NULL},
  {"_cuda_getDevice",   (PyCFunction)THCPModule_getDevice_wrap,   METH_NOARGS,  NULL},
  {"_cuda_getDeviceCount", (PyCFunction)THCPModule_getDeviceCount_wrap, METH_NOARGS, NULL},
  {"_cuda_getDeviceName", (PyCFunction)THCPModule_getDeviceName_wrap, METH_O,   NULL},
  {"_cuda_getDeviceCapability", (PyCFunction)THCPModule_getDeviceCapability_wrap, METH_O,   NULL},
  {"_cuda_getCurrentStream", (PyCFunction)THCPModule_getCurrentStream_wrap, METH_NOARGS, NULL},
  {"_cuda_getCurrentBlasHandle", (PyCFunction)THCPModule_getCurrentBlasHandle_wrap, METH_NOARGS, NULL},
  {"_cuda_setStream",    (PyCFunction)THCPModule_setStream_wrap,  METH_O, NULL},
  {"_cuda_isDriverSufficient", (PyCFunction)THCPModule_isDriverSufficient, METH_NOARGS, NULL},
  {"_cuda_getDriverVersion", (PyCFunction)THCPModule_getDriverVersion, METH_NOARGS, NULL},
  {"_cuda_getCompiledVersion", (PyCFunction)THCPModule_getCompiledVersion, METH_NOARGS, NULL},
  {"_cuda_getRNGState", (PyCFunction)THCPModule_getRNGState,      METH_NOARGS,  NULL},
  {"_cuda_setRNGState", (PyCFunction)THCPModule_setRNGState,      METH_O,       NULL},
  {"_cuda_emptyCache", (PyCFunction) THCPModule_emptyCache,       METH_NOARGS,  NULL},
  {"_cuda_manualSeed",  (PyCFunction)THCPModule_manualSeed,       METH_O,       NULL},
  {"_cuda_manualSeedAll", (PyCFunction)THCPModule_manualSeedAll,  METH_O,       NULL},
  {"_cuda_seed",        (PyCFunction)THCPModule_seed,             METH_NOARGS,  NULL},
  {"_cuda_seedAll",     (PyCFunction)THCPModule_seedAll,          METH_NOARGS,  NULL},
  {"_cuda_initialSeed", (PyCFunction)THCPModule_initialSeed,      METH_NOARGS,  NULL},
  {"_cuda_cudaHostAllocator", (PyCFunction)THCPModule_cudaHostAllocator, METH_NOARGS, NULL},
  {"_cuda_synchronize", (PyCFunction)THCPModule_cudaSynchronize, METH_NOARGS, NULL},
  {"_cuda_sleep", (PyCFunction)THCPModule_cudaSleep, METH_O, NULL},
  {"_cuda_lock_mutex",   (PyCFunction)THCPModule_cudaLockMutex,   METH_NOARGS,  NULL},
  {"_cuda_unlock_mutex", (PyCFunction)THCPModule_cudaUnlockMutex, METH_NOARGS,  NULL},
#ifdef WITH_NCCL
  {"_nccl_version", (PyCFunction)THCPModule_nccl_version, METH_NOARGS, NULL},
  {"_nccl_unique_id", (PyCFunction)THCPModule_nccl_unique_id, METH_NOARGS, NULL},
  {"_nccl_init_rank", (PyCFunction)THCPModule_nccl_init_rank, METH_VARARGS, NULL},
  {"_nccl_reduce", (PyCFunction)THCPModule_nccl_reduce, METH_VARARGS, NULL},
  {"_nccl_all_reduce", (PyCFunction)THCPModule_nccl_all_reduce, METH_VARARGS, NULL},
  {"_nccl_broadcast", (PyCFunction)THCPModule_nccl_broadcast, METH_VARARGS, NULL},
  {"_nccl_all_gather", (PyCFunction)THCPModule_nccl_all_gather, METH_VARARGS, NULL},
  {"_nccl_reduce_scatter", (PyCFunction)THCPModule_nccl_reduce_scatter, METH_VARARGS, NULL},
#endif
  {NULL}
};

PyMethodDef* THCPModule_methods() {
  return _THCPModule_methods;
}
