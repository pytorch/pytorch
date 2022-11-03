#include <ATen/ATen.h>
#include <ATen/cuda/CUDAConfig.h>
#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>

#endif
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/cuda/Sleep.h>
#include <ATen/cuda/detail/CUDAHooks.h>
#include <ATen/cuda/jiterator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#ifdef USE_NCCL
#include <torch/csrc/cuda/python_nccl.h>
#endif
#include <c10/util/CallOnce.h>
#include <c10/util/irange.h>

#include <torch/csrc/CudaIPCTypes.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/cuda/THCP.h>
#include <torch/csrc/cuda/python_comm.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

#include <array>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>
#include <unordered_map>

#ifndef WIN32
#include <pthread.h>
#endif

using namespace torch;

static bool in_bad_fork = false; // True for children forked after cuda init

#ifndef WIN32
// Called in the forked child if cuda has already been initialized
static void forked_child() {
  in_bad_fork = true;
  torch::utils::set_requires_cuda_init(true);
}
#endif

// Should be called before the first cuda call.
// Note: This is distinct from initExtension because a stub cuda implementation
// has some working functions (e.g. device_count) but cannot fully initialize.
static void poison_fork() {
#ifndef WIN32
  static c10::once_flag flag;
  c10::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
#endif
}

////////////////////////////////////////////////////////////////////////////////
// CUDA management methods
////////////////////////////////////////////////////////////////////////////////

void THCPModule_setDevice(int device) {
  c10::cuda::set_device(static_cast<c10::DeviceIndex>(device));
}

PyObject* THCPModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to setDevice");
  int64_t device = THPUtils_unpackLong(arg);

  torch::utils::cuda_lazy_init();
  THCPModule_setDevice(device);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::cuda_lazy_init();
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto device = static_cast<int>(c10::cuda::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_canDeviceAccessPeer_wrap(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* arg1 = nullptr;
  PyObject* arg2 = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "can_device_peer_access",
        1,
        "(int device, int peer_device);");
    return nullptr;
  }
  THPUtils_assert(
      THPUtils_checkLong(arg1), "invalid argument to canDeviceAccessPeer");
  THPUtils_assert(
      THPUtils_checkLong(arg2), "invalid argument to canDeviceAccessPeer");
  int64_t device = THPUtils_unpackLong(arg1);
  int64_t peer_device = THPUtils_unpackLong(arg2);

  torch::utils::cuda_lazy_init();
  auto can_access = at::cuda::canDeviceAccessPeer(device, peer_device);
  return PyBool_FromLong(can_access);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  poison_fork();
  return THPUtils_packUInt64(at::cuda::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getArchFlags(PyObject* self, PyObject* noargs) {
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

static PyObject* THCPModule_isInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getCurrentStream_wrap(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromUnsignedLongLong(
      at::cuda::getCurrentCUDAStream(device).pack());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getCurrentStream_raw(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromVoidPtr(at::cuda::getCurrentCUDAStream(device).stream());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getDefaultStream_wrap(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromUnsignedLongLong(
      at::cuda::getDefaultCUDAStream(device).pack());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_setStream_wrap(PyObject* self, PyObject* obj) {
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

PyObject* THCPModule_getCompiledVersion(PyObject* self, PyObject* noargs) {
#if defined(USE_ROCM)
  return THPUtils_packInt64((int64_t)ROCM_VERSION);
#else
  return THPUtils_packInt64((int64_t)CUDA_VERSION);
#endif
}

PyObject* THCPModule_cudaHostAllocator(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::Allocator* allocator = at::cuda::getCachingHostAllocator();
  return PyLong_FromVoidPtr(allocator);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cudaCachingAllocator_raw_alloc(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* size_o = nullptr;
  PyObject* stream_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &size_o, &stream_o)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "caching_allocator_alloc",
        1,
        "(ssize_t size, intptr_t stream);");
    return nullptr;
  }
  auto size = PyLong_AsSsize_t(size_o);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  cudaStream_t stream = static_cast<cudaStream_t>(PyLong_AsVoidPtr(stream_o));
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  void* mem =
      c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(size, stream);
  return PyLong_FromVoidPtr(mem);
  END_HANDLE_TH_ERRORS
}

// Unpack a PyObject to at::Scalar, throw an exception if it fails
at::Scalar as_scalar(PyObject* arg) {
  // Zero-dim tensors are converted to Scalars as-is. Note this doesn't
  // currently handle most NumPy scalar types except np.float64.
  if (THPVariable_Check(arg)) {
    return THPVariable_Unpack(arg).item();
  }

  if (THPUtils_checkLong(arg)) {
    return at::Scalar(static_cast<int64_t>(THPUtils_unpackLong(arg)));
  }

  if (PyBool_Check(arg)) {
    return at::Scalar(THPUtils_unpackBool(arg));
  }

  if (PyComplex_Check(arg)) {
    return at::Scalar(THPUtils_unpackComplexDouble(arg));
  }
  return at::Scalar(THPUtils_unpackDouble(arg));
}

// Entrypoint for the callable created by torch.cuda.jiterator
// See jiterator.py for more details
PyObject* THCPModule_cudaJiteratorCompileAndLaunchKernel(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS

  PyObject* code_string_o = nullptr;
  PyObject* kernel_name_o = nullptr;
  PyObject* return_by_ref_o = nullptr;
  PyObject* num_outputs_o = nullptr;
  PyObject* tensors_o = nullptr;
  PyObject* kwargs_o = nullptr;
  if (!PyArg_ParseTuple(
          args,
          "OOOOO|O",
          &code_string_o,
          &kernel_name_o,
          &return_by_ref_o,
          &num_outputs_o,
          &tensors_o,
          &kwargs_o)) {
    return nullptr;
  }

  const std::string code_string = THPUtils_unpackString(code_string_o);
  const std::string kernel_name = THPUtils_unpackString(kernel_name_o);
  const bool return_by_ref = THPUtils_unpackBool(return_by_ref_o);
  const int num_outputs = static_cast<int>(THPUtils_unpackLong(num_outputs_o));

  THPUtils_assert(
      PyTuple_Check(tensors_o),
      "tensors argument is expected to "
      "be a tuple, but got %s",
      THPUtils_typename(tensors_o));
  Py_ssize_t num_tensors = PyTuple_GET_SIZE(tensors_o);

  c10::SmallVector<at::Tensor> tensors;
  for (const auto i : c10::irange(num_tensors)) {
    PyObject* _tensor = PyTuple_GET_ITEM(tensors_o, i);
    THPUtils_assert(
        THPVariable_Check(_tensor),
        "%d of input tensors tuple is not a Tensor",
        i);

    tensors.emplace_back(THPVariable_Unpack(_tensor));
  }

  c10::SmallVector<at::Scalar> extra_args;
  PyObject* key = nullptr;
  PyObject* value = nullptr;
  Py_ssize_t pos = 0;
  while (PyDict_Next(kwargs_o, &pos, &key, &value)) {
    extra_args.emplace_back(as_scalar(value));
  }

  c10::SmallVector<at::Tensor> outputs = at::cuda::CompileAndLaunchKernel(
      code_string,
      kernel_name,
      num_outputs,
      tensors,
      extra_args,
      return_by_ref);

  if (num_outputs == 1) {
    return THPVariable_Wrap(outputs[0]);
  } else {
    PyObject* output_tuple = PyTuple_New(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      PyTuple_SetItem(output_tuple, i, THPVariable_Wrap(outputs[i]));
    }
    return output_tuple;
  }

  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cudaCachingAllocator_raw_delete(
    PyObject* _unused,
    PyObject* obj) {
  HANDLE_TH_ERRORS
  void* mem_ptr = PyLong_AsVoidPtr(obj);
  c10::cuda::CUDACachingAllocator::raw_delete(mem_ptr);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cudaCachingAllocator_set_allocator_settings(
    PyObject* _unused,
    PyObject* env) {
  HANDLE_TH_ERRORS
  c10::cuda::CUDACachingAllocator::setAllocatorSettings(
      THPUtils_unpackString(env));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getAllocatorBackend(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(c10::cuda::CUDACachingAllocator::name());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cudaSynchronize(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::cuda::device_synchronize();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cudaIPCCollect(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::CudaIPCCollect();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cudaSleep(PyObject* _unused, PyObject* cycles) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(cycles), "torch.cuda._sleep(): expected 'int'");
  at::cuda::sleep(THPUtils_unpackLong(cycles));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// We need to ensure that as long as a thread will NEVER loose the GIL as long
// as it holds the CUDA mutex. Otherwise another thread might be scheduled and
// try to e.g. allocate a new tensor which will cause a deadlock. It's enough to
// have a single global, because it can be only set once (cudaMutex is not
// recursive) by the thread that owns the mutex (obviously there can be only one
// such thread).
static PyGILState_STATE cudaMutexGILState;

PyObject* THCPModule_cudaLockMutex(PyObject* module, PyObject* noargs) {
  auto mutex = c10::cuda::getFreeMutex();
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

PyObject* THCPModule_cudaUnlockMutex(PyObject* module, PyObject* noargs) {
  auto mutex = c10::cuda::getFreeMutex();
  PyGILState_Release(cudaMutexGILState);
  mutex->unlock();
  Py_RETURN_NONE;
}

PyObject* THCPModule_hasPrimaryContext(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to has_primary_context");
  int64_t device_index = static_cast<int64_t>(THPUtils_unpackLong(arg));
  if (at::cuda::detail::hasPrimaryContext(device_index)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_setMemoryFraction(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* fraction_o = nullptr;
  PyObject* device_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &fraction_o, &device_o)) {
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

PyObject* THCPModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::cuda::CUDACachingAllocator::emptyCache();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THCPModule_memoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  const int device = (int)THPUtils_unpackLong(arg);

  using c10::cuda::CUDACachingAllocator::DeviceStats;
  using c10::cuda::CUDACachingAllocator::Stat;
  using c10::cuda::CUDACachingAllocator::StatArray;
  using c10::cuda::CUDACachingAllocator::StatType;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
        statTypeNames = {"all", "small_pool", "large_pool"};
    py::dict dict;
    for (const auto i : c10::irange(statTypeNames.size())) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const DeviceStats stats =
      c10::cuda::CUDACachingAllocator::getDeviceStats(device);

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

PyObject* THCPModule_resetAccumulatedMemoryStats(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg),
      "invalid argument to reset_accumulated_memory_stats");
  const int device = (int)THPUtils_unpackLong(arg);
  c10::cuda::CUDACachingAllocator::resetAccumulatedStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THCPModule_resetPeakMemoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  const int device = (int)THPUtils_unpackLong(arg);
  c10::cuda::CUDACachingAllocator::resetPeakStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

struct Frame {
  PyCodeObject* code;
  int lasti;
};

struct StackContext : public c10::cuda::CUDACachingAllocator::Context {
  std::vector<Frame> frames;
  // Empty if cpp traces weren't enabled
  std::string cpp_frames;
  ~StackContext() {
    py::gil_scoped_acquire acquire;
    for (auto& f : frames) {
      Py_XDECREF((PyObject*)f.code);
    }
  }
  static std::shared_ptr<StackContext> _gather() {
    py::gil_scoped_acquire acquire;
    auto r = std::make_shared<StackContext>();
    PyFrameObject* f = PyEval_GetFrame();
    Py_XINCREF(f);
    while (f) {
      r->frames.emplace_back(Frame{PyFrame_GetCode(f), PyFrame_GetLasti(f)});
      auto f_back = PyFrame_GetBack(f);
      Py_XDECREF(f);
      f = f_back;
    }
    return r;
  }
  static std::shared_ptr<c10::cuda::CUDACachingAllocator::Context> gather() {
    return _gather();
  }
  static std::shared_ptr<c10::cuda::CUDACachingAllocator::Context>
  gather_with_cpp() {
    auto r = _gather();
    r->cpp_frames = c10::get_backtrace();
    return std::move(r);
  }
};

PyObject* THCPModule_memorySnapshot(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS

  using c10::cuda::CUDACachingAllocator::BlockInfo;
  using c10::cuda::CUDACachingAllocator::History;
  using c10::cuda::CUDACachingAllocator::SegmentInfo;

  py::str device_s = "device";
  py::str address_s = "address";
  py::str total_size_s = "total_size";
  py::str allocated_size_s = "allocated_size";
  py::str active_size_s = "active_size";
  py::str stream_s = "stream";
  py::str segment_type_s = "segment_type";
  py::str large_s = "large";
  py::str small_s = "small";
  py::str size_s = "size";
  py::str state_s = "state";
  py::str active_allocated_s = "active_allocated";
  py::str active_pending_free_s = "active_pending_free";
  py::str inactive_s = "inactive";
  py::str addr_s = "addr";
  py::str real_size_s = "real_size";
  py::str filename_s = "filename";
  py::str name_s = "name";
  py::str line_s = "line";
  py::str frames_s = "frames";
  py::str cpp_frames_s = "cpp_frames";
  py::str history_s = "history";
  py::str blocks_s = "blocks";

  std::unordered_map<StackContext*, py::list> cached_frames;
  const auto get_frames = [&](StackContext* sc) -> py::list {
    auto it = cached_frames.find(sc);
    if (it != cached_frames.end()) {
      return it->second;
    }
    py::list frames;
    for (auto& f : sc->frames) {
      py::dict frame;
      frame[filename_s] =
          py::reinterpret_borrow<py::object>(f.code->co_filename);
      frame[name_s] = py::reinterpret_borrow<py::object>(f.code->co_name);
      frame[line_s] = PyCode_Addr2Line(f.code, f.lasti);
      frames.append(std::move(frame));
    }
    cached_frames.insert({sc, frames});
    return frames;
  };

  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict[device_s] = segmentInfo.device;
    segmentDict[address_s] = segmentInfo.address;
    segmentDict[total_size_s] = segmentInfo.total_size;
    segmentDict[allocated_size_s] = segmentInfo.allocated_size;
    segmentDict[active_size_s] = segmentInfo.active_size;
    // we want the python objects to pickle easily so use an int to
    // represent the stream rather than a torch.cuda.stream object
    segmentDict[stream_s] = int64_t(segmentInfo.stream);
    segmentDict[segment_type_s] = (segmentInfo.is_large ? large_s : small_s);

    py::list blocks;
    for (const auto& blockInfo : segmentInfo.blocks) {
      py::dict blockDict;
      blockDict[size_s] = blockInfo.size;
      blockDict[state_s] =
          (blockInfo.allocated
               ? active_allocated_s
               : (blockInfo.active ? active_pending_free_s : inactive_s));
      if (blockInfo.history.size()) {
        py::list history;
        for (const History& h : blockInfo.history) {
          py::dict history_entry;
          history_entry[addr_s] = (int64_t)h.addr;
          history_entry[real_size_s] = h.real_size;
          if (h.context) {
            auto sc = (StackContext*)h.context.get();
            history_entry[frames_s] = get_frames(sc);
            if (!sc->cpp_frames.empty()) {
              history_entry[cpp_frames_s] = py::cast(sc->cpp_frames);
            }
          }
          history.append(std::move(history_entry));
        }
        blockDict[history_s] = std::move(history);
      }
      blocks.append(blockDict);
    }
    segmentDict[blocks_s] = blocks;

    return segmentDict;
  };

  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  py::list segments;

  for (const auto& segmentInfo : snapshot.segments) {
    segments.append(segmentInfoToDict(segmentInfo));
  }

  py::list traces;
  py::str action_s = "action";
  py::str alloc_s = "alloc";
  py::str free_requested_s = "free_requested";
  py::str free_completed_s = "free_completed";
  py::str segment_alloc_s = "segment_alloc";
  py::str segment_free_s = "segment_free";
  py::str snapshot_s = "snapshot";
  py::str oom_s = "oom";
  py::str device_free_s = "device_free";

  using namespace c10::cuda::CUDACachingAllocator;

  auto action_to_str = [&](TraceEntry::Action action) {
    switch (action) {
      case TraceEntry::ALLOC:
        return alloc_s;
      case TraceEntry::FREE_REQUESTED:
        return free_requested_s;
      case TraceEntry::FREE_COMPLETED:
        return free_completed_s;
      case TraceEntry::SEGMENT_ALLOC:
        return segment_alloc_s;
      case TraceEntry::SEGMENT_FREE:
        return segment_free_s;
      case TraceEntry::OOM:
        return oom_s;
      case TraceEntry::SNAPSHOT:
        return snapshot_s;
    }
    throw std::runtime_error("unreachable");
  };

  for (const auto& traceInfo : snapshot.device_traces) {
    py::list trace;
    for (const auto& te : traceInfo) {
      py::dict trace_entry;
      if (te.context_) {
        // without further compression frames can get really large on dump
        auto sc = (StackContext*)te.context_.get();
        trace_entry[frames_s] = get_frames(sc);
        if (!sc->cpp_frames.empty()) {
          trace_entry[cpp_frames_s] = py::cast(sc->cpp_frames);
        }
      }
      trace_entry[action_s] = action_to_str(te.action_);
      trace_entry[TraceEntry::OOM == te.action_ ? device_free_s : addr_s] =
          te.addr_;
      trace_entry[size_s] = te.size_;
      trace_entry[stream_s] = int64_t(te.stream_);
      trace.append(trace_entry);
    }
    traces.append(trace);
  }

  py::dict result;
  result["segments"] = segments;
  result["device_traces"] = traces;

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_attachOutOfMemoryObserver(
    PyObject* _unused,
    PyObject* observer) {
  HANDLE_TH_ERRORS
  Py_XINCREF(observer);
  auto obs = [observer](
                 int64_t device,
                 int64_t alloc,
                 int64_t device_allocated,
                 int64_t device_free) {
    py::gil_scoped_acquire g;
    PyObject* result = PyObject_CallFunction(
        observer, "LLLL", device, alloc, device_allocated, device_free);
    if (!result) {
      throw py::error_already_set();
    }
    Py_XDECREF(result);
  };
  c10::cuda::CUDACachingAllocator::attachOutOfMemoryObserver(std::move(obs));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cudaSetSyncDebugMode(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_WARN_ONCE(
      "Synchronization debug mode is a prototype feature and does not yet detect all "
      "synchronizing operations");
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to set_sync_debug_mode");
  int64_t debug_mode = THPUtils_unpackLong(arg);
  TORCH_CHECK(
      debug_mode >= 0 && debug_mode <= 2,
      "invalid value of debug_mode, expected one of 0,1,2");
  c10::cuda::SyncDebugMode l;
  switch (debug_mode) {
    case 0:
      l = c10::cuda::SyncDebugMode::L_DISABLED;
      break;
    case 1:
      l = c10::cuda::SyncDebugMode::L_WARN;
      break;
    case 2:
      l = c10::cuda::SyncDebugMode::L_ERROR;
      break;
    default:
      l = c10::cuda::SyncDebugMode::L_DISABLED;
      break; // can't happen
  }
  c10::cuda::warning_state().set_sync_debug_mode(l);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_cudaGetSyncDebugMode(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto debug_mode = c10::cuda::warning_state().get_sync_debug_mode();
  switch (debug_mode) {
    case c10::cuda::SyncDebugMode::L_DISABLED:
      return THPUtils_packInt32(0);
    case c10::cuda::SyncDebugMode::L_WARN:
      return THPUtils_packInt32(1);
    case c10::cuda::SyncDebugMode::L_ERROR:
      return THPUtils_packInt32(2);
    default:
      return THPUtils_packInt32(-1); // can't happen
  }
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
      .def_readonly(
          "multi_processor_count", &cudaDeviceProp::multiProcessorCount)
      .def_readonly("total_memory", &cudaDeviceProp::totalGlobalMem)
      .def("__repr__", [](const cudaDeviceProp& prop) {
        std::ostringstream stream;
        stream << "_CudaDeviceProperties(name='" << prop.name
               << "', major=" << prop.major << ", minor=" << prop.minor
               << ", total_memory=" << prop.totalGlobalMem / (1024 * 1024)
               << "MB, multi_processor_count=" << prop.multiProcessorCount
               << ")";
        return stream.str();
      });

  m.def(
      "_cuda_recordMemoryHistory",
      [](bool enabled,
         bool record_context,
         bool record_context_cpp,
         Py_ssize_t alloc_trace_max_entries,
         bool alloc_trace_record_context) {
        c10::cuda::CUDACachingAllocator::recordHistory(
            enabled,
            record_context ? (record_context_cpp ? StackContext::gather_with_cpp
                                                 : StackContext::gather)
                           : nullptr,
            alloc_trace_max_entries,
            alloc_trace_record_context);
      });
}

static void bindGetDeviceProperties(PyObject* module) {
  // Add method to torch.cuda
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_get_device_properties",
      [](int device) -> cudaDeviceProp* {
        return at::cuda::getDeviceProperties(device);
      },
      py::return_value_policy::reference);
}

// Callback for python part. Used for additional initialization of python
// classes
static PyObject* THCPModule_initExtension(PyObject* self, PyObject* noargs) {
#if C10_ASAN_ENABLED
  TORCH_WARN(
      "torch.cuda: your pytorch binary has address sanitizer (asan) built in, "
      "asan is currently not compatible with torch.cuda module, "
      "you might get unexpected behavior (eg. out of memory, crash, etc.), "
      "please rebuild pytorch without asan if you need to use this module");
#endif
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!in_bad_fork); // Handled at python level
  poison_fork();
  at::globalContext().lazyInitCUDA();

  auto m = THPObjectPtr(PyImport_ImportModule("torch.cuda"));
  if (!m)
    throw python_error();

  bool has_half = true;

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  set_module_attr("has_magma", at::hasMAGMA() ? Py_True : Py_False);
  set_module_attr("has_half", has_half ? Py_True : Py_False);

  auto num_gpus = c10::cuda::device_count();
  auto default_cuda_generators = PyTuple_New(static_cast<Py_ssize_t>(num_gpus));
  for (const auto i : c10::irange(num_gpus)) {
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

PyObject* THCPModule_getCurrentBlasHandle_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  return PyLong_FromVoidPtr(handle);
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPModule_clearBlasWorkspaces_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::cuda::clearCublasWorkspaces();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_rocm_is_backward_pass(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
#if USE_ROCM
  if (at::ROCmBackwardPassGuard::is_backward_pass()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
#else
  Py_RETURN_FALSE;
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPModule_isCurrentStreamCapturing_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // If there's no cuda context, at::cuda::currentStreamCaptureStatus returns
  // CaptureStatus::None without initializing a context.
  if (at::cuda::currentStreamCaptureStatus() == at::cuda::CaptureStatus::None) {
    Py_RETURN_FALSE;
  } else {
    Py_RETURN_TRUE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_setBenchmarkLimitCuDNN(PyObject* _unused, PyObject* arg) {
  THPUtils_assert(
      THPUtils_checkLong(arg),
      "set_benchmark_limit_cudnn expects an int, "
      "but got %s",
      THPUtils_typename(arg));
  auto benchmark_limit = static_cast<int>(THPUtils_unpackLong(arg));
#if defined(USE_ROCM)
  TORCH_WARN_ONCE(
      "cuDNN Benchmark limit is not supported in MIOpen and will have no effect.");
#endif
#if AT_CUDNN_ENABLED()
#if HAS_CUDNN_V8()
  at::globalContext().setBenchmarkLimitCuDNN(benchmark_limit);
#else
  TORCH_WARN_ONCE(
      "cuDNN Benchmark limit is not supported with cuDNN v7 API and will have no effect.");
#endif
#endif
  Py_RETURN_NONE;
}

PyObject* THCPModule_benchmarkLimitCuDNN(PyObject* _unused, PyObject* noargs) {
  return THPUtils_packInt32(at::globalContext().benchmarkLimitCuDNN());
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-avoid-c-arrays)
static struct PyMethodDef _THCPModule_methods[] = {
    {"_cuda_init", THCPModule_initExtension, METH_NOARGS, nullptr},
    {"_cuda_setDevice", THCPModule_setDevice_wrap, METH_O, nullptr},
    {"_cuda_getDevice", THCPModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_cuda_getDeviceCount",
     THCPModule_getDeviceCount_wrap,
     METH_NOARGS,
     nullptr},
    {"_cuda_canDeviceAccessPeer",
     THCPModule_canDeviceAccessPeer_wrap,
     METH_VARARGS,
     nullptr},
    {"_cuda_getArchFlags", THCPModule_getArchFlags, METH_NOARGS, nullptr},
    {"_cuda_isInBadFork", THCPModule_isInBadFork, METH_NOARGS, nullptr},
    {"_cuda_getCurrentStream",
     THCPModule_getCurrentStream_wrap,
     METH_O,
     nullptr},
    {"_cuda_getCurrentRawStream",
     THCPModule_getCurrentStream_raw,
     METH_O,
     nullptr},
    {"_cuda_getDefaultStream",
     THCPModule_getDefaultStream_wrap,
     METH_O,
     nullptr},
    {"_cuda_getCurrentBlasHandle",
     THCPModule_getCurrentBlasHandle_wrap,
     METH_NOARGS,
     nullptr},
    {"_cuda_clearCublasWorkspaces",
     THCPModule_clearBlasWorkspaces_wrap,
     METH_NOARGS,
     nullptr},
    {"_cuda_isCurrentStreamCapturing",
     THCPModule_isCurrentStreamCapturing_wrap,
     METH_NOARGS,
     nullptr},
    {"_cuda_setStream", THCPModule_setStream_wrap, METH_O, nullptr},
    {"_cuda_getCompiledVersion",
     THCPModule_getCompiledVersion,
     METH_NOARGS,
     nullptr},
    {"_cuda_hasPrimaryContext", THCPModule_hasPrimaryContext, METH_O, nullptr},
    {"_cuda_setMemoryFraction",
     THCPModule_setMemoryFraction,
     METH_VARARGS,
     nullptr},
    {"_cuda_emptyCache", THCPModule_emptyCache, METH_NOARGS, nullptr},
    {"_cuda_memoryStats", THCPModule_memoryStats, METH_O, nullptr},
    {"_cuda_resetAccumulatedMemoryStats",
     THCPModule_resetAccumulatedMemoryStats,
     METH_O,
     nullptr},
    {"_cuda_resetPeakMemoryStats",
     THCPModule_resetPeakMemoryStats,
     METH_O,
     nullptr},
    {"_cuda_memorySnapshot", THCPModule_memorySnapshot, METH_NOARGS, nullptr},
    {"_cuda_attach_out_of_memory_observer",
     THCPModule_attachOutOfMemoryObserver,
     METH_O,
     nullptr},
    {"_cuda_cudaHostAllocator",
     THCPModule_cudaHostAllocator,
     METH_NOARGS,
     nullptr},
    {"_cuda_cudaCachingAllocator_raw_alloc",
     THCPModule_cudaCachingAllocator_raw_alloc,
     METH_VARARGS,
     nullptr},
    {"_cuda_cudaCachingAllocator_raw_delete",
     THCPModule_cudaCachingAllocator_raw_delete,
     METH_O,
     nullptr},
    {"_cuda_cudaCachingAllocator_set_allocator_settings",
     THCPModule_cudaCachingAllocator_set_allocator_settings,
     METH_O,
     nullptr},
    {"_cuda_getAllocatorBackend",
     THCPModule_getAllocatorBackend,
     METH_NOARGS,
     nullptr},
    {"_cuda_synchronize", THCPModule_cudaSynchronize, METH_NOARGS, nullptr},
    {"_cuda_ipc_collect", THCPModule_cudaIPCCollect, METH_NOARGS, nullptr},
    {"_cuda_sleep", THCPModule_cudaSleep, METH_O, nullptr},
    {"_cuda_lock_mutex", THCPModule_cudaLockMutex, METH_NOARGS, nullptr},
    {"_cuda_unlock_mutex", THCPModule_cudaUnlockMutex, METH_NOARGS, nullptr},
    {"_cuda_set_sync_debug_mode",
     THCPModule_cudaSetSyncDebugMode,
     METH_O,
     nullptr},
    {"_cuda_get_sync_debug_mode",
     THCPModule_cudaGetSyncDebugMode,
     METH_NOARGS,
     nullptr},
    {"_cuda_jiterator_compile_and_launch_kernel",
     THCPModule_cudaJiteratorCompileAndLaunchKernel,
     METH_VARARGS,
     nullptr},
    {"_cuda_get_cudnn_benchmark_limit",
     THCPModule_benchmarkLimitCuDNN,
     METH_NOARGS,
     nullptr},
    {"_cuda_set_cudnn_benchmark_limit",
     THCPModule_setBenchmarkLimitCuDNN,
     METH_O,
     nullptr},
#ifdef USE_NCCL
    {"_nccl_version", THCPModule_nccl_version, METH_NOARGS, nullptr},
    {"_nccl_unique_id", THCPModule_nccl_unique_id, METH_NOARGS, nullptr},
    {"_nccl_init_rank", THCPModule_nccl_init_rank, METH_VARARGS, nullptr},
    {"_nccl_reduce", THCPModule_nccl_reduce, METH_VARARGS, nullptr},
    {"_nccl_all_reduce", THCPModule_nccl_all_reduce, METH_VARARGS, nullptr},
    {"_nccl_broadcast", THCPModule_nccl_broadcast, METH_VARARGS, nullptr},
    {"_nccl_all_gather", THCPModule_nccl_all_gather, METH_VARARGS, nullptr},
    {"_nccl_reduce_scatter",
     THCPModule_nccl_reduce_scatter,
     METH_VARARGS,
     nullptr},
#endif
    {"_rocm_is_backward_pass",
     THCPModule_rocm_is_backward_pass,
     METH_NOARGS,
     nullptr},
    {nullptr}};

PyMethodDef* THCPModule_methods() {
  return _THCPModule_methods;
}

namespace torch {
namespace cuda {

namespace shared {

void initCudartBindings(PyObject* module);
void initNvtxBindings(PyObject* module);
#if defined(USE_CUDNN) || defined(USE_ROCM)
void initCudnnBindings(PyObject* module);
#endif

} // namespace shared

void initModule(PyObject* module) {
  python::initCommMethods(module);
  // As weird as it seems, this file is also compiled for ROCm,
  // so this condition might not always be true...
  shared::initCudartBindings(module);
  shared::initNvtxBindings(module);
#if defined(USE_CUDNN) || defined(USE_ROCM)
  shared::initCudnnBindings(module);
#endif
  registerCudaDeviceProperties(module);
}

} // namespace cuda
} // namespace torch
