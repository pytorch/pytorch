#include "torch/csrc/python_headers.h"

#include <stdbool.h>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <sstream>
#include <TH/TH.h>
#include <ATen/ATen.h>
#include "ATen/cuda/CUDAContext.h"
#include <THC/THCCachingAllocator.h>
#ifdef USE_NCCL
#include <nccl.h>
#endif

#include "THCP.h"

#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/cuda/python_comm.h"

using namespace torch;

THCState *state;

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
  using namespace at;
  using namespace torch::autograd;
  HANDLE_TH_ERRORS
  Variable var = VariableType::getType(CPU(kByte))->tensor();
  THCRandom_getRNGState(state, (THByteTensor*)(var.data().unsafeGetTensorImpl()));
  return THPVariable_Wrap(var);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_setRNGState(PyObject *_unused, PyObject *obj)
{
  HANDLE_TH_ERRORS
  if (!THPVariable_Check(obj) || THPVariable_UnpackData(obj).type().ID() != at::TypeID::CPUByte) {
    throw TypeError("set_rng_state expects a torch.ByteTensor, but got %s",
        Py_TYPE(obj)->tp_name);
  }
  auto& tensor = THPVariable_UnpackData(obj);
  THCRandom_setRNGState(state, (THByteTensor*)tensor.unsafeGetTensorImpl());
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
  return THPUtils_packUInt64(THCRandom_seed(state));
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_seedAll(PyObject *_unused)
{
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(THCRandom_seedAll(state));
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_initialSeed(PyObject *_unused)
{
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(THCRandom_initialSeed(state));
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
  THCCachingAllocator_emptyCache();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject * THCPModule_memoryAllocated(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  int device = (int) THPUtils_unpackLong(arg);
  auto memory_allocated = THCCachingAllocator_currentMemoryAllocated(device);
  return PyLong_FromUnsignedLongLong(memory_allocated);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_maxMemoryAllocated(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to max_memory_allocated");
  int device = (int) THPUtils_unpackLong(arg);
  auto max_memory_allocated = THCCachingAllocator_maxMemoryAllocated(device);
  return PyLong_FromUnsignedLongLong(max_memory_allocated);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_memoryCached(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to memory_cached");
  int device = (int) THPUtils_unpackLong(arg);
  auto memory_cached = THCCachingAllocator_currentMemoryCached(device);
  return PyLong_FromUnsignedLongLong(memory_cached);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_maxMemoryCached(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to max_memory_cached");
  int device = (int) THPUtils_unpackLong(arg);
  auto max_memory_cached = THCCachingAllocator_maxMemoryCached(device);
  return PyLong_FromUnsignedLongLong(max_memory_cached);
  END_HANDLE_TH_ERRORS
}

////////////////////////////////////////////////////////////////////////////////
// Cuda module initialization
////////////////////////////////////////////////////////////////////////////////

static void bindCudaDeviceProperties(PyObject* module) {
  // Add class and method to torch.cuda
  auto m = py::handle(module).cast<py::module>();
  py::class_<cudaDeviceProp>(m, "_CudaDeviceProperties")
    .def_readonly("name", &cudaDeviceProp::name)
    .def_readonly("major", &cudaDeviceProp::major)
    .def_readonly("minor", &cudaDeviceProp::minor)
    .def_readonly("is_multi_gpu_board", &cudaDeviceProp::isMultiGpuBoard)
    .def_readonly("is_integrated", &cudaDeviceProp::integrated)
    .def_readonly("multi_processor_count", &cudaDeviceProp::multiProcessorCount)
    .def_readonly("total_memory", &cudaDeviceProp::totalGlobalMem)
    .def("__repr__", [](const cudaDeviceProp &prop) {
      std::ostringstream stream;
      stream << "_CudaDeviceProperties(name='" << prop.name << "', major=" << prop.major
             << ", minor=" << prop.minor << ", total_memory=" << prop.totalGlobalMem / (1024 * 1024)
             << "MB, multi_processor_count=" << prop.multiProcessorCount << ")";
      return stream.str();
    });
  m.def("_get_device_properties", [](int device) -> cudaDeviceProp * {
    return at::cuda::getDeviceProperties(device);
  }, py::return_value_policy::reference);
}

// Callback for python part. Used for additional initialization of python classes
static PyObject * THCPModule_initExtension(PyObject *self)
{
  HANDLE_TH_ERRORS
  state = at::globalContext().lazyInitCUDA();

  auto m = THPObjectPtr(PyImport_ImportModule("torch.cuda"));
  if (!m) throw python_error();

  // Register Storage Python objects with DynamicTypes.cpp
  THCPDoubleStorage_postInit(m);
  THCPFloatStorage_postInit(m);
  THCPHalfStorage_postInit(m);
  THCPLongStorage_postInit(m);
  THCPIntStorage_postInit(m);
  THCPShortStorage_postInit(m);
  THCPCharStorage_postInit(m);
  THCPByteStorage_postInit(m);

#ifdef USE_MAGMA
  THCMagma_init(state);
  bool has_magma = true;
#else
  bool has_magma = false;
#endif

  bool has_half = true;

  auto set_module_attr = [&](const char* name, PyObject* v) {
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  set_module_attr("has_magma", has_magma ? Py_True : Py_False);
  set_module_attr("has_half", has_half ? Py_True : Py_False);

  auto _state_cdata = THPObjectPtr(PyLong_FromVoidPtr(state));
  if (!_state_cdata) throw python_error();
  set_module_attr("_state_cdata", _state_cdata.get());

  bindCudaDeviceProperties(m);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

#ifdef USE_NCCL
#include "python_nccl.h"

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
  {"_cuda_getCurrentStream", (PyCFunction)THCPModule_getCurrentStream_wrap, METH_NOARGS, NULL},
  {"_cuda_getCurrentBlasHandle", (PyCFunction)THCPModule_getCurrentBlasHandle_wrap, METH_NOARGS, NULL},
  {"_cuda_setStream",    (PyCFunction)THCPModule_setStream_wrap,  METH_O, NULL},
  {"_cuda_isDriverSufficient", (PyCFunction)THCPModule_isDriverSufficient, METH_NOARGS, NULL},
  {"_cuda_getDriverVersion", (PyCFunction)THCPModule_getDriverVersion, METH_NOARGS, NULL},
  {"_cuda_getCompiledVersion", (PyCFunction)THCPModule_getCompiledVersion, METH_NOARGS, NULL},
  {"_cuda_getRNGState", (PyCFunction)THCPModule_getRNGState,      METH_NOARGS,  NULL},
  {"_cuda_setRNGState", (PyCFunction)THCPModule_setRNGState,      METH_O,       NULL},
  {"_cuda_emptyCache", (PyCFunction) THCPModule_emptyCache,       METH_NOARGS,  NULL},
  {"_cuda_memoryAllocated", (PyCFunction) THCPModule_memoryAllocated, METH_O,  NULL},
  {"_cuda_maxMemoryAllocated", (PyCFunction) THCPModule_maxMemoryAllocated, METH_O,  NULL},
  {"_cuda_memoryCached", (PyCFunction) THCPModule_memoryCached, METH_O,  NULL},
  {"_cuda_maxMemoryCached", (PyCFunction) THCPModule_maxMemoryCached, METH_O,  NULL},
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
#ifdef USE_NCCL
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

namespace torch { namespace cuda {

void initModule(PyObject *module) {
  python::initCommMethods(module);
}

}}
