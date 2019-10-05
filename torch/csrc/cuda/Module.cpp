#include <torch/csrc/python_headers.h>

#include <array>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <sstream>
#include <TH/TH.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/CUDAGenerator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#ifdef USE_NCCL
#include <nccl.h>
#endif

#include <torch/csrc/cuda/THCP.h>
#include <torch/csrc/CudaIPCTypes.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/utils/python_dicts.h>
#include <torch/csrc/utils/python_lists.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/cuda/python_comm.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/Generator.h>

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

  torch::utils::cuda_lazy_init();
  THCPModule_setDevice(device);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_getDevice_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  int device;
  torch::utils::cuda_lazy_init();
  THCudaCheck(cudaGetDevice(&device));
  return PyLong_FromLong(device);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_getDeviceCount_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  //torch::utils::cuda_lazy_init();
  return PyLong_FromLong(at::cuda::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_set_run_yet_variable_to_false_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  torch::utils::set_run_yet_variable_to_false();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_getCurrentStream_wrap(
    PyObject * /* unused */, PyObject *device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
    THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromUnsignedLongLong(
    at::cuda::getCurrentCUDAStream(device).pack());
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_getDefaultStream_wrap(
    PyObject * /* unused */, PyObject *device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
    THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromUnsignedLongLong(
    at::cuda::getDefaultCUDAStream(device).pack());
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_setStream_wrap(PyObject *self, PyObject *obj)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(PyLong_Check(obj), "invalid stream");
  uint64_t bits = PyLong_AsUnsignedLongLong(obj);
  if (bits == static_cast<uint64_t>(-1) && PyErr_Occurred()) {
    throw python_error();
  }
  auto stream = at::cuda::CUDAStream::unpack(bits);
  int device;
  THCudaCheck(cudaGetDevice(&device));
  if (device != stream.device_index()) {
    THCPModule_setDevice(stream.device_index());
  }
  at::cuda::setCurrentCUDAStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_isDriverSufficient(PyObject *self, PyObject *noargs)
{
  int count;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err == cudaErrorInsufficientDriver) {
    return PyBool_FromLong(0);
  }
  return PyBool_FromLong(1);
}

PyObject * THCPModule_getDriverVersion(PyObject *self, PyObject *noargs)
{
  int driverVersion = -1;
  cudaError_t err = cudaDriverGetVersion(&driverVersion);
  if (err != cudaSuccess) {
    PyErr_Format(PyExc_RuntimeError,
                    "Error calling cudaDriverGetVersion: %d %s",
                    err, cudaGetErrorString(err));
    return nullptr;
  }
  return PyLong_FromLong((int64_t) driverVersion);
}

PyObject * THCPModule_getCompiledVersion(PyObject *self, PyObject *noargs)
{
  return PyLong_FromLong((long) CUDA_VERSION);
}

PyObject * THCPModule_cudaHostAllocator(PyObject *_unused, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  c10::Allocator* allocator = THCState_getCudaHostAllocator(state);
  return PyLong_FromVoidPtr(allocator);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_cudaSynchronize(PyObject *_unused, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  THCudaCheck(cudaDeviceSynchronize());
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_cudaIPCCollect(PyObject *_unused, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  torch::CudaIPCCollect();
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

PyObject * THCPModule_cudaLockMutex(PyObject *module, PyObject *noargs)
{
  auto mutex = c10::cuda::CUDACachingAllocator::getFreeMutex();
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

PyObject * THCPModule_cudaUnlockMutex(PyObject *module, PyObject *noargs)
{
  auto mutex = c10::cuda::CUDACachingAllocator::getFreeMutex();
  PyGILState_Release(cudaMutexGILState);
  mutex->unlock();
  Py_RETURN_NONE;
}

PyObject * THCPModule_hasPrimaryContext(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to has_primary_context");
  int64_t device_index = static_cast<int64_t>(THPUtils_unpackLong(arg));
  if (at::detail::getCUDAHooks().hasPrimaryContext(device_index)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_emptyCache(PyObject *_unused, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  c10::cuda::CUDACachingAllocator::emptyCache();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject * THCPModule_memoryStats(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  const int device = (int) THPUtils_unpackLong(arg);

  using c10::cuda::CUDACachingAllocator::StatType;
  using c10::cuda::CUDACachingAllocator::Stat;
  using c10::cuda::CUDACachingAllocator::StatArray;
  using c10::cuda::CUDACachingAllocator::DeviceStats;

  const auto statToDict = [=](const Stat& stat) {
    PyObject* const dict = THPUtils_newDict();
    THPUtils_setDictStrInt64(dict, "current", stat.current);
    THPUtils_setDictStrInt64(dict, "peak", stat.peak);
    THPUtils_setDictStrInt64(dict, "allocated", stat.allocated);
    THPUtils_setDictStrInt64(dict, "freed", stat.freed);
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)> statTypeNames = {
      "all", "small_pool", "large_pool"
    };
    PyObject* const dict = THPUtils_newDict();
    for (size_t i = 0; i < statTypeNames.size(); ++i) {
      THPUtils_setDictStrPyObject(dict, statTypeNames[i], statToDict(statArray[i]));
    }
    return dict;
  };

  const auto setDictStatArray = [=](PyObject* dict, const char* name, const StatArray& statArray) {
    THPUtils_setDictStrPyObject(dict, name, statArrayToDict(statArray));
  };

  const DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
  PyObject* const result = THPUtils_newDict();

  THPUtils_setDictStrInt64(result, "num_alloc_retries", stats.num_alloc_retries);
  THPUtils_setDictStrInt64(result, "num_ooms", stats.num_ooms);

  setDictStatArray(result, "allocation", stats.allocation);
  setDictStatArray(result, "segment", stats.segment);
  setDictStatArray(result, "active", stats.active);
  setDictStatArray(result, "inactive_split", stats.inactive_split);
  setDictStatArray(result, "allocated_bytes", stats.allocated_bytes);
  setDictStatArray(result, "reserved_bytes", stats.reserved_bytes);
  setDictStatArray(result, "active_bytes", stats.active_bytes);
  setDictStatArray(result, "inactive_split_bytes", stats.inactive_split_bytes);

  return result;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_resetAccumulatedMemoryStats(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to reset_accumulated_memory_stats");
  const int device = (int) THPUtils_unpackLong(arg);
  c10::cuda::CUDACachingAllocator::resetAccumulatedStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject * THCPModule_resetPeakMemoryStats(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  const int device = (int) THPUtils_unpackLong(arg);
  c10::cuda::CUDACachingAllocator::resetPeakStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject * THCPModule_memorySnapshot(PyObject *_unused, PyObject *noargs)
{
  HANDLE_TH_ERRORS

  using c10::cuda::CUDACachingAllocator::SegmentInfo;
  using c10::cuda::CUDACachingAllocator::BlockInfo;

  const std::vector<SegmentInfo>& snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  PyObject* const result = THPUtils_newList();

  const auto segmentInfoToDict = [](const SegmentInfo& segmentInfo) {
    PyObject* const segmentDict = THPUtils_newDict();

    THPUtils_setDictStrInt64(segmentDict, "device", segmentInfo.device);
    THPUtils_setDictStrInt64(segmentDict, "address", segmentInfo.address);
    THPUtils_setDictStrInt64(segmentDict, "total_size", segmentInfo.total_size);
    THPUtils_setDictStrInt64(segmentDict, "allocated_size", segmentInfo.allocated_size);
    THPUtils_setDictStrInt64(segmentDict, "active_size", segmentInfo.active_size);
    THPUtils_setDictStrStr(segmentDict, "segment_type", segmentInfo.is_large ? "large" : "small");

    PyObject* const blocks = THPUtils_newList();
    for (const auto& blockInfo : segmentInfo.blocks) {
      PyObject* const blockDict = THPUtils_newDict();
      THPUtils_setDictStrInt64(blockDict, "size", blockInfo.size);
      THPUtils_setDictStrStr(blockDict, "state", (blockInfo.allocated ? "active_allocated" : (blockInfo.active ? "active_pending_free" : "inactive")));
      THPUtils_appendListPyObject(blocks, blockDict);
    }
    THPUtils_setDictStrPyObject(segmentDict, "blocks", blocks);

    return segmentDict;
  };

  for (const auto& segmentInfo : snapshot) {
    THPUtils_appendListPyObject(result, segmentInfoToDict(segmentInfo));
  }

  return result;
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
static PyObject * THCPModule_initExtension(PyObject *self, PyObject *noargs)
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
  THCPBoolStorage_postInit(m);
  THCPBFloat16Storage_postInit(m);

  bool has_half = true;

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  set_module_attr("has_magma", at::hasMAGMA() ? Py_True : Py_False);
  set_module_attr("has_half", has_half ? Py_True : Py_False);

  auto _state_cdata = THPObjectPtr(PyLong_FromVoidPtr(state));
  if (!_state_cdata) throw python_error();
  set_module_attr("_state_cdata", _state_cdata.get());

  auto num_gpus = c10::cuda::device_count();
  auto default_cuda_generators = PyTuple_New(static_cast<Py_ssize_t>(num_gpus));
  for(int i = 0; i < num_gpus; i++) {
    auto gen = at::cuda::detail::getDefaultCUDAGenerator(i);
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_cuda_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_cuda_generators);

  bindCudaDeviceProperties(m);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

#ifdef USE_NCCL
#include <torch/csrc/cuda/python_nccl.h>

void THCPModule_useNccl()
{
  // Use NCCL to ensure that the symbols are loaded
  ncclUniqueId uniqueId;
  ncclGetUniqueId(&uniqueId);
}
#endif

PyObject * THCPModule_getCurrentBlasHandle_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  return PyLong_FromVoidPtr(handle);
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef _THCPModule_methods[] = {
  {"_cuda_init",        (PyCFunction)THCPModule_initExtension,    METH_NOARGS,  nullptr},
  {"_cuda_setDevice",   (PyCFunction)THCPModule_setDevice_wrap,   METH_O,       nullptr},
  {"_cuda_getDevice",   (PyCFunction)THCPModule_getDevice_wrap,   METH_NOARGS,  nullptr},
  {"_cuda_getDeviceCount", (PyCFunction)THCPModule_getDeviceCount_wrap, METH_NOARGS, nullptr},
  {"_cuda_set_run_yet_variable_to_false",
    (PyCFunction)THCPModule_set_run_yet_variable_to_false_wrap, METH_NOARGS, nullptr},
  {"_cuda_getCurrentStream",
    (PyCFunction)THCPModule_getCurrentStream_wrap, METH_O, nullptr},
  {"_cuda_getDefaultStream",
    (PyCFunction)THCPModule_getDefaultStream_wrap, METH_O, nullptr},
  {"_cuda_getCurrentBlasHandle", (PyCFunction)THCPModule_getCurrentBlasHandle_wrap, METH_NOARGS, nullptr},
  {"_cuda_setStream",    (PyCFunction)THCPModule_setStream_wrap,  METH_O, nullptr},
  {"_cuda_isDriverSufficient", (PyCFunction)THCPModule_isDriverSufficient, METH_NOARGS, nullptr},
  {"_cuda_getDriverVersion", (PyCFunction)THCPModule_getDriverVersion, METH_NOARGS, nullptr},
  {"_cuda_getCompiledVersion", (PyCFunction)THCPModule_getCompiledVersion, METH_NOARGS, nullptr},
  {"_cuda_hasPrimaryContext", (PyCFunction) THCPModule_hasPrimaryContext,  METH_O,  nullptr},
  {"_cuda_emptyCache", (PyCFunction) THCPModule_emptyCache, METH_NOARGS, nullptr},
  {"_cuda_memoryStats", (PyCFunction) THCPModule_memoryStats, METH_O, nullptr},
  {"_cuda_resetAccumulatedMemoryStats", (PyCFunction) THCPModule_resetAccumulatedMemoryStats, METH_O, nullptr},
  {"_cuda_resetPeakMemoryStats", (PyCFunction) THCPModule_resetPeakMemoryStats, METH_O,  nullptr},
  {"_cuda_memorySnapshot", (PyCFunction) THCPModule_memorySnapshot, METH_NOARGS, nullptr},
  {"_cuda_cudaHostAllocator", (PyCFunction)THCPModule_cudaHostAllocator, METH_NOARGS, nullptr},
  {"_cuda_synchronize", (PyCFunction)THCPModule_cudaSynchronize, METH_NOARGS, nullptr},
  {"_cuda_ipc_collect", (PyCFunction)THCPModule_cudaIPCCollect, METH_NOARGS, nullptr},
  {"_cuda_sleep", (PyCFunction)THCPModule_cudaSleep, METH_O, nullptr},
  {"_cuda_lock_mutex",   (PyCFunction)THCPModule_cudaLockMutex,   METH_NOARGS,  nullptr},
  {"_cuda_unlock_mutex", (PyCFunction)THCPModule_cudaUnlockMutex, METH_NOARGS,  nullptr},
#ifdef USE_NCCL
  {"_nccl_version", (PyCFunction)THCPModule_nccl_version, METH_NOARGS, nullptr},
  {"_nccl_unique_id", (PyCFunction)THCPModule_nccl_unique_id, METH_NOARGS, nullptr},
  {"_nccl_init_rank", (PyCFunction)THCPModule_nccl_init_rank, METH_VARARGS, nullptr},
  {"_nccl_reduce", (PyCFunction)THCPModule_nccl_reduce, METH_VARARGS, nullptr},
  {"_nccl_all_reduce", (PyCFunction)THCPModule_nccl_all_reduce, METH_VARARGS, nullptr},
  {"_nccl_broadcast", (PyCFunction)THCPModule_nccl_broadcast, METH_VARARGS, nullptr},
  {"_nccl_all_gather", (PyCFunction)THCPModule_nccl_all_gather, METH_VARARGS, nullptr},
  {"_nccl_reduce_scatter", (PyCFunction)THCPModule_nccl_reduce_scatter, METH_VARARGS, nullptr},
#endif
  {nullptr}
};

PyMethodDef* THCPModule_methods() {
  return _THCPModule_methods;
}

namespace torch { namespace cuda {

void initModule(PyObject *module) {
  python::initCommMethods(module);
}

}}
