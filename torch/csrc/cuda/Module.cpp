#include <TH/TH.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/detail/CUDAHooks.h>
#ifdef USE_NCCL
#include <torch/csrc/cuda/python_nccl.h>
#endif
#include <c10/util/irange.h>

#include <torch/csrc/cuda/THCP.h>
#include <torch/csrc/CudaIPCTypes.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/cuda/python_comm.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/python_headers.h>

#include <array>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <sstream>

#ifndef WIN32
#include <pthread.h>
#endif

using namespace torch;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
THCState *state = nullptr;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static bool in_bad_fork = false;  // True for children forked after cuda init

#ifndef WIN32
// Called in the forked child if cuda has already been initialized
static void forked_child() {
  in_bad_fork = true;
  torch::utils::set_run_yet_variable_to_false();
  state = nullptr;
}
#endif

// Should be called before the first cuda call.
// Note: This is distinct from initExtension because a stub cuda implementation
// has some working functions (e.g. device_count) but cannot fully initialize.
static void poison_fork() {
#ifndef WIN32
  static std::once_flag flag;
  std::call_once(flag, []{ pthread_atfork(nullptr, nullptr, forked_child); });
#endif
}

////////////////////////////////////////////////////////////////////////////////
// CUDA management methods
////////////////////////////////////////////////////////////////////////////////

void THCPModule_setDevice(int device)
{
  c10::cuda::set_device(static_cast<c10::DeviceIndex>(device));
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
  torch::utils::cuda_lazy_init();
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto device = static_cast<int>(c10::cuda::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_canDeviceAccessPeer_wrap(PyObject *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* arg1 = nullptr;
  PyObject* arg2 = nullptr;
  if(!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "can_device_peer_access",
        1,
        "(int device, int peer_device);");
    return nullptr;
  }
  THPUtils_assert(THPUtils_checkLong(arg1), "invalid argument to canDeviceAccessPeer");
  THPUtils_assert(THPUtils_checkLong(arg2), "invalid argument to canDeviceAccessPeer");
  int64_t device = THPUtils_unpackLong(arg1);
  int64_t peer_device = THPUtils_unpackLong(arg2);

  torch::utils::cuda_lazy_init();
  auto can_access = at::cuda::canDeviceAccessPeer(device, peer_device);
  return PyBool_FromLong(can_access);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_getDeviceCount_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  poison_fork();
  return THPUtils_packUInt64(at::cuda::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_getArchFlags(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  poison_fork();
#ifdef CUDA_ARCH_FLAGS
  static const char* flags = C10_STRINGIZE(CUDA_ARCH_FLAGS);
  return THPUtils_packString(flags);
#else
  Py_RETURN_NONE;
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject * THCPModule_isInBadFork(PyObject *self, PyObject *noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
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
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto device = static_cast<int>(c10::cuda::current_device());
  if (device != stream.device_index()) {
    THCPModule_setDevice(stream.device_index());
  }
  at::cuda::setCurrentCUDAStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_getCompiledVersion(PyObject *self, PyObject *noargs)
{
  return THPUtils_packInt64((int64_t) CUDA_VERSION);
}

PyObject * THCPModule_cudaHostAllocator(PyObject *_unused, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  c10::Allocator* allocator = THCState_getCudaHostAllocator(state);
  return PyLong_FromVoidPtr(allocator);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_cudaCachingAllocator_raw_alloc(PyObject *_unused, PyObject *args){
  HANDLE_TH_ERRORS
  PyObject* size_o = nullptr;
  PyObject* stream_o = nullptr;
  if(!PyArg_ParseTuple(args, "OO", &size_o, &stream_o)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "caching_allocator_alloc",
        1,
        "(ssize_t size, intptr_t stream);");
    return nullptr;
  }
  ssize_t size = PyLong_AsSsize_t(size_o);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  cudaStream_t stream = static_cast<cudaStream_t>(PyLong_AsVoidPtr(stream_o));
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  void* mem = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(size, stream);
  return PyLong_FromVoidPtr(mem);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_cudaCachingAllocator_raw_delete(PyObject *_unused, PyObject *obj){
  HANDLE_TH_ERRORS
  void* mem_ptr = PyLong_AsVoidPtr(obj);
  c10::cuda::CUDACachingAllocator::raw_delete(mem_ptr);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_cudaSynchronize(PyObject *_unused, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  c10::cuda::device_synchronize();
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

// We need to ensure that as long as a thread will NEVER loose the GIL as long
// as it holds the CUDA mutex. Otherwise another thread might be scheduled and
// try to e.g. allocate a new tensor which will cause a deadlock. It's enough to
// have a single global, because it can be only set once (cudaMutex is not
// recursive) by the thread that owns the mutex (obviously there can be only one
// such thread).
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
      pybind11::gil_scoped_release no_gil;
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
  if (at::cuda::detail::hasPrimaryContext(device_index)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_setMemoryFraction(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* fraction_o = nullptr;
  PyObject* device_o = nullptr;
  if(!PyArg_ParseTuple(args, "OO", &fraction_o, &device_o)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "set_memory_fraction",
        1,
        "(double fraction, int device);");
    return nullptr;
  }
  double fraction = PyFloat_AsDouble(fraction_o);
  int64_t device = PyLong_AsLongLong(device_o);

  c10::cuda::CUDACachingAllocator::setMemoryFraction(fraction, device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
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

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)> statTypeNames = {
      "all", "small_pool", "large_pool"
    };
    py::dict dict;
    for (const auto i : c10::irange(statTypeNames.size())) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);

  py::dict result;
  result["num_alloc_retries"] = stats.num_alloc_retries;
  result["num_ooms"] = stats.num_ooms;
  result["max_split_size"] = stats.max_split_size;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["oversize_allocations"] = statToDict(stats.oversize_allocations);
  result["oversize_segments"] = statToDict(stats.oversize_segments);

  return result.release().ptr();
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

  const auto segmentInfoToDict = [](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict["device"] = segmentInfo.device;
    segmentDict["address"] = segmentInfo.address;
    segmentDict["total_size"] = segmentInfo.total_size;
    segmentDict["allocated_size"] = segmentInfo.allocated_size;
    segmentDict["active_size"] = segmentInfo.active_size;
    segmentDict["segment_type"] = (segmentInfo.is_large ? "large" : "small");

    py::list blocks;
    for (const auto& blockInfo : segmentInfo.blocks) {
      py::dict blockDict;
      blockDict["size"] = blockInfo.size;
      blockDict["state"] = (blockInfo.allocated ? "active_allocated" : (blockInfo.active ? "active_pending_free" : "inactive"));
      blocks.append(blockDict);
    }
    segmentDict["blocks"] = blocks;

    return segmentDict;
  };

  const std::vector<SegmentInfo>& snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  py::list result;

  for (const auto& segmentInfo : snapshot) {
    result.append(segmentInfoToDict(segmentInfo));
  }

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

////////////////////////////////////////////////////////////////////////////////
// Cuda module initialization
////////////////////////////////////////////////////////////////////////////////

static void registerCudaDeviceProperties(PyObject* module) {
  // Add _cudaDevicePropertires class to torch._C
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
}

static void bindGetDeviceProperties(PyObject* module) {
  // Add method to torch.cuda
  auto m = py::handle(module).cast<py::module>();
  m.def("_get_device_properties", [](int device) -> cudaDeviceProp * {
    return at::cuda::getDeviceProperties(device);
  }, py::return_value_policy::reference);
}

// Callback for python part. Used for additional initialization of python classes
static PyObject * THCPModule_initExtension(PyObject *self, PyObject *noargs)
{
#if C10_ASAN_ENABLED
  TORCH_WARN(
    "torch.cuda: your pytorch binary has address sanitizer (asan) built in, "
    "asan is currently not compatible with torch.cuda module, "
    "you might get unexpected behavior (eg. out of memory, crash, etc.), "
    "please rebuild pytorch without asan if you need to use this module");
#endif
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!in_bad_fork);  // Handled at python level
  poison_fork();
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
  THCPComplexDoubleStorage_postInit(m);
  THCPComplexFloatStorage_postInit(m);

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
  for(const auto i : c10::irange(num_gpus)) {
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto gen = at::cuda::detail::getDefaultCUDAGenerator(i);
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_cuda_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_cuda_generators);
  bindGetDeviceProperties(m);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_getCurrentBlasHandle_wrap(PyObject *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  return PyLong_FromVoidPtr(handle);
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays, cppcoreguidelines-avoid-non-const-global-variables, cppcoreguidelines-avoid-c-arrays)
static struct PyMethodDef _THCPModule_methods[] = {
  {"_cuda_init",        THCPModule_initExtension,    METH_NOARGS,  nullptr},
  {"_cuda_setDevice",   THCPModule_setDevice_wrap,   METH_O,       nullptr},
  {"_cuda_getDevice",   THCPModule_getDevice_wrap,   METH_NOARGS,  nullptr},
  {"_cuda_getDeviceCount", THCPModule_getDeviceCount_wrap, METH_NOARGS, nullptr},
  {"_cuda_canDeviceAccessPeer", THCPModule_canDeviceAccessPeer_wrap, METH_VARARGS, nullptr},
  {"_cuda_getArchFlags", THCPModule_getArchFlags, METH_NOARGS, nullptr},
  {"_cuda_isInBadFork", THCPModule_isInBadFork, METH_NOARGS, nullptr},
  {"_cuda_getCurrentStream",
    THCPModule_getCurrentStream_wrap, METH_O, nullptr},
  {"_cuda_getDefaultStream",
    THCPModule_getDefaultStream_wrap, METH_O, nullptr},
  {"_cuda_getCurrentBlasHandle", THCPModule_getCurrentBlasHandle_wrap, METH_NOARGS, nullptr},
  {"_cuda_setStream",    THCPModule_setStream_wrap,  METH_O, nullptr},
  {"_cuda_getCompiledVersion", THCPModule_getCompiledVersion, METH_NOARGS, nullptr},
  {"_cuda_hasPrimaryContext", THCPModule_hasPrimaryContext,  METH_O,  nullptr},
  {"_cuda_setMemoryFraction", THCPModule_setMemoryFraction, METH_VARARGS,  nullptr},
  {"_cuda_emptyCache", THCPModule_emptyCache, METH_NOARGS, nullptr},
  {"_cuda_memoryStats", THCPModule_memoryStats, METH_O, nullptr},
  {"_cuda_resetAccumulatedMemoryStats", THCPModule_resetAccumulatedMemoryStats, METH_O, nullptr},
  {"_cuda_resetPeakMemoryStats", THCPModule_resetPeakMemoryStats, METH_O,  nullptr},
  {"_cuda_memorySnapshot", THCPModule_memorySnapshot, METH_NOARGS, nullptr},
  {"_cuda_cudaHostAllocator", THCPModule_cudaHostAllocator, METH_NOARGS, nullptr},
  {"_cuda_cudaCachingAllocator_raw_alloc", THCPModule_cudaCachingAllocator_raw_alloc, METH_VARARGS, nullptr},
  {"_cuda_cudaCachingAllocator_raw_delete", THCPModule_cudaCachingAllocator_raw_delete, METH_O, nullptr},
  {"_cuda_synchronize", THCPModule_cudaSynchronize, METH_NOARGS, nullptr},
  {"_cuda_ipc_collect", THCPModule_cudaIPCCollect, METH_NOARGS, nullptr},
  {"_cuda_sleep", THCPModule_cudaSleep, METH_O, nullptr},
  {"_cuda_lock_mutex",   THCPModule_cudaLockMutex,   METH_NOARGS,  nullptr},
  {"_cuda_unlock_mutex", THCPModule_cudaUnlockMutex, METH_NOARGS,  nullptr},
#ifdef USE_NCCL
  {"_nccl_version", THCPModule_nccl_version, METH_NOARGS, nullptr},
  {"_nccl_unique_id", THCPModule_nccl_unique_id, METH_NOARGS, nullptr},
  {"_nccl_init_rank", THCPModule_nccl_init_rank, METH_VARARGS, nullptr},
  {"_nccl_reduce", THCPModule_nccl_reduce, METH_VARARGS, nullptr},
  {"_nccl_all_reduce", THCPModule_nccl_all_reduce, METH_VARARGS, nullptr},
  {"_nccl_broadcast", THCPModule_nccl_broadcast, METH_VARARGS, nullptr},
  {"_nccl_all_gather", THCPModule_nccl_all_gather, METH_VARARGS, nullptr},
  {"_nccl_reduce_scatter", THCPModule_nccl_reduce_scatter, METH_VARARGS, nullptr},
#endif
  {nullptr}
};

PyMethodDef* THCPModule_methods() {
  return _THCPModule_methods;
}

namespace torch { namespace cuda {

namespace shared {

void initCudartBindings(PyObject* module);
void initNvtxBindings(PyObject* module);
#if defined(USE_CUDNN) || defined(__HIP_PLATFORM_HCC__)
void initCudnnBindings(PyObject* module);
#endif

} // namespace shared

void initModule(PyObject *module) {
  python::initCommMethods(module);
  // As weird as it seems, this file is also compiled for ROCm,
  // so this condition might not always be true...
  shared::initCudartBindings(module);
  shared::initNvtxBindings(module);
#if defined(USE_CUDNN) || defined(__HIP_PLATFORM_HCC__)
  shared::initCudnnBindings(module);
#endif
  registerCudaDeviceProperties(module);
}

}}
