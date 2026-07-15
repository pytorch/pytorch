#include <torch/csrc/python_headers.h>
#ifdef _MSC_VER
#include <c10/util/win32-headers.h>
#endif
#include <structmember.h>

#include <c10/core/CPUAllocator.h>
#include <libshm.h>
#include <torch/csrc/CudaIPCTypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/copy_utils.h>

#include <c10/util/intrusive_ptr.h>
#include <fmt/format.h>

#include <torch/csrc/Storage.h>
#include <torch/csrc/StorageSharing.h>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef USE_XPU
#include <ATen/detail/XPUHooksInterface.h>
#include <ATen/xpu/level_zero_stub/ATenLevelZero.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#endif

#include <ATen/MapAllocator.h>
#include <ATen/StorageUtils.h>
#include <torch/csrc/utils/python_numbers.h>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

static PyObject* THPStorage_sharedDecref(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(self);
  c10::DeviceType device_type = storage.device_type();
  if (device_type == at::kCPU) {
    THManagedMapAllocator* ctx =
        THManagedMapAllocator::fromDataPtr(storage.data_ptr());
    if (ctx) {
      ctx->decref();
    }
  }
  return Py_NewRef(self);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_sharedIncref(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(self);
  c10::DeviceType device_type = storage.device_type();
  if (device_type == at::kCPU) {
    THManagedMapAllocator* ctx =
        THManagedMapAllocator::fromDataPtr(storage.data_ptr());
    if (ctx) {
      ctx->incref();
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_pyNewFilenameStorage(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  long long size = 0;
  if (!PyArg_ParseTuple(args, "L", &size)) {
    return nullptr;
  }
  if (size < 0) {
    return nullptr;
  }

  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
  std::string handle = at::NewProcessWideShmHandle();
  return THPStorage_NewWithStorage(
      THPStorageClass,
      c10::make_intrusive<at::StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size,
          THManagedMapAllocator::makeDataPtr(
              "", handle.c_str(), flags, static_cast<size_t>(size)),
          /*allocator=*/nullptr,
          /*resizable=*/false));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_shareFilename(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(self);
  TORCH_CHECK(
      storage.device_type() == at::kCPU,
      "_share_filename_: only available on CPU");
  THManagedMapAllocator* ctx =
      THManagedMapAllocator::fromDataPtr(storage.data_ptr());
  // Storage is already in shared memory, just return a handle
  if (ctx) {
    // done
  } else {
    // TODO: retry on collision
    // TODO: free GIL - but remember to reacquire it when an exception is thrown
    int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
    std::string handle = at::NewProcessWideShmHandle();
    // Create a new storage in shared memory
    at::Storage new_storage(c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        storage.nbytes(),
        THManagedMapAllocator::makeDataPtr(
            "", handle.c_str(), flags, storage.nbytes()),
        /*allocator=*/nullptr,
        /*resizable=*/false));

    {
      // Copying into shared memory can be slow, so release the GIL
      pybind11::gil_scoped_release no_gil;
      // Copy data from old storage into the new one
      at::storage_copy(new_storage, storage);
    }

    // Replace the old data_ptr and allocator with the new ones
    storage.set_data_ptr(std::move(new_storage.mutable_data_ptr()));
    storage.unsafeGetStorageImpl()->set_allocator(new_storage.allocator());

    ctx = THManagedMapAllocator::fromDataPtr(storage.data_ptr());
    AT_ASSERT(ctx);
  }

  THPObjectPtr manager_handle(PyBytes_FromString(ctx->manager_handle()));
  if (!manager_handle)
    return nullptr;
  THPObjectPtr storage_handle(PyBytes_FromString(ctx->filename()));
  if (!storage_handle)
    return nullptr;
  THPObjectPtr size(THPUtils_packUInt64(storage.nbytes()));
  if (!size)
    return nullptr;

  THPObjectPtr tuple(PyTuple_New(3));
  if (!tuple)
    return nullptr;
  PyTuple_SET_ITEM(tuple.get(), 0, manager_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 1, storage_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 2, size.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_newSharedFilename(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 3, "tuple of 3 items expected");
  PyObject* _manager_handle = PyTuple_GET_ITEM(args, 0);
  PyObject* _object_handle = PyTuple_GET_ITEM(args, 1);
  PyObject* _size = PyTuple_GET_ITEM(args, 2);
  if (!PyBytes_Check(_manager_handle) || !PyBytes_Check(_object_handle) ||
      !THPUtils_checkLong(_size)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_new_shared in file system mode",
        1,
        "a handle (string/bytes) and storage size (int)");
    return nullptr;
  }
  const char* manager_handle = PyBytes_AS_STRING(_manager_handle);
  const char* object_handle = PyBytes_AS_STRING(_object_handle);
  uint64_t size = THPUtils_unpackUInt64(_size);
  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
  return THPStorage_NewWithStorage(
      THPStorageClass,
      c10::make_intrusive<at::StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size,
          THManagedMapAllocator::makeDataPtr(
              manager_handle, object_handle, flags, size),
          /*allocator=*/nullptr,
          /*resizable=*/false));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_pyNewFdStorage(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  long long size = 0;
  if (!PyArg_ParseTuple(args, "L", &size)) {
    return nullptr;
  }
  if (size < 0) {
    return nullptr;
  }
  return THPStorage_NewWithStorage(
      THPStorageClass, at::new_shm_fd_storage(size));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_shareFd(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(self);
  TORCH_CHECK(
      storage.device_type() == at::kCPU, "_share_fd_: only available on CPU");
  at::MapAllocator* ctx = at::MapAllocator::fromDataPtr(storage.data_ptr());
  // Storage is already in shared memory, just return a handle
  if (ctx) {
    // done
  } else {
    at::Storage new_storage(at::new_shm_fd_storage(storage.nbytes()));
    {
      // Copying into shared memory can be slow, so release the GIL
      pybind11::gil_scoped_release no_gil;
      // Copy data from old storage into the new one
      at::storage_copy(new_storage, storage);
    }

    // Replace the old data_ptr and allocator with the new ones
    storage.set_data_ptr(std::move(new_storage.mutable_data_ptr()));
    storage.unsafeGetStorageImpl()->set_allocator(new_storage.allocator());

    ctx = at::MapAllocator::fromDataPtr(storage.data_ptr());
    AT_ASSERT(ctx);
  }

  THPObjectPtr storage_handle(THPUtils_packInt32(ctx->fd()));
  if (!storage_handle)
    return nullptr;
  THPObjectPtr size(THPUtils_packUInt64(storage.nbytes()));
  if (!size)
    return nullptr;

  THPObjectPtr tuple(PyTuple_New(2));
  if (!tuple)
    return nullptr;
  PyTuple_SET_ITEM(tuple.get(), 0, storage_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 1, size.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_newSharedFd(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 2, "tuple of 2 items expected");
  PyObject* _tmp_fd = PyTuple_GET_ITEM(args, 0);
  PyObject* _size = PyTuple_GET_ITEM(args, 1);
  if (!THPUtils_checkLong(_tmp_fd) || !THPUtils_checkLong(_size)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_new_shared in file descriptor mode",
        1,
        "a file descriptor (int) and storage size (int)");
    return nullptr;
  }
  int tmp_fd = THPUtils_unpackInt(_tmp_fd);
  int64_t size = THPUtils_unpackLong(_size);
  int fd = dup(tmp_fd);
  if (fd == -1) {
    THPUtils_setError("could not duplicate a shared memory file descriptor");
    return nullptr;
  }

  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE |
      at::ALLOCATOR_MAPPED_KEEPFD | at::ALLOCATOR_MAPPED_FROMFD;
  return THPStorage_NewWithStorage(
      THPStorageClass,
      c10::make_intrusive<at::StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          size,
          at::MapAllocator::makeDataPtr(
              at::WITH_FD, "", fd, flags, size, nullptr),
          /*allocator=*/nullptr,
          /*resizable=*/false));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_shareCuda(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
#ifdef USE_CUDA
  const auto& storage = THPStorage_Unpack(self);
  TORCH_CHECK(
      storage.device_type() == at::kCUDA,
      "_share_cuda_: only available on CUDA");
  c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();

  if (storage_impl->received_cuda()) {
    TORCH_CHECK(
        false,
        "Attempted to send CUDA tensor received from another process; this is not currently supported. Consider cloning before sending.");
  }

  at::DeviceGuard device_guard(storage.device());
  THPObjectPtr tuple(PyTuple_New(8));
  THPObjectPtr device(THPUtils_packInt32(storage.device().index()));
  THPObjectPtr _handle(Py_NewRef(Py_None));
  THPObjectPtr size_bytes(THPUtils_packUInt64(storage.nbytes()));
  THPObjectPtr _offset_bytes(THPUtils_packInt32(0));
  THPObjectPtr _ref_counter(Py_NewRef(Py_None));
  THPObjectPtr _ref_counter_offset(THPUtils_packInt32(0));
  THPObjectPtr _event_handle(Py_NewRef(Py_None));
  THPObjectPtr _event_sync_required(Py_NewRef(Py_None));
  if (storage.data()) {
    auto shandle =
        c10::cuda::CUDACachingAllocator::shareIpcHandle(storage.mutable_data());
    _handle = PyBytes_FromStringAndSize(
        shandle.handle.c_str(), static_cast<Py_ssize_t>(shandle.handle.size()));
    _offset_bytes = PyLong_FromSsize_t(static_cast<Py_ssize_t>(shandle.offset));

    // Put Storage Data behind new ref counting context
    // See Note [CUDA IPC Refcounting implementation explained]
    at::DataPtr sent_data_ptr = torch::GetNewRefCountedSentData(
        storage.mutable_data(), storage.device());
    auto old_data_ptr = storage.set_data_ptr(std::move(sent_data_ptr));
    auto sent_data =
        static_cast<torch::CudaIPCSentData*>(storage.data_ptr().get_context());
    sent_data->set_original_ptr(std::move(old_data_ptr));
    _ref_counter = PyBytes_FromString((sent_data->handle()).c_str());
    _ref_counter_offset = THPUtils_packUInt64(sent_data->offset());

    cudaIpcEventHandle_t ipc_event_handle{};

    if (sent_data->event_sync_required_) {
      C10_CUDA_CHECK(
          cudaIpcGetEventHandle(&ipc_event_handle, sent_data->event_));
    }

    _event_handle = PyBytes_FromStringAndSize(
        reinterpret_cast<const char*>(&ipc_event_handle), CUDA_IPC_HANDLE_SIZE);
    _event_sync_required = PyBool_FromLong(sent_data->event_sync_required_);
  }

  if (!tuple || !device || !_handle || !size_bytes || !_offset_bytes ||
      !_event_handle) {
    return nullptr;
  }
  PyTuple_SET_ITEM(tuple.get(), 0, device.release());
  // cudaIpcMemHandle_t(of basePtr)
  PyTuple_SET_ITEM(tuple.get(), 1, _handle.release());
  // Size(in bytes) of the real storage, note this is not the size of basePtr
  // memory block.
  PyTuple_SET_ITEM(tuple.get(), 2, size_bytes.release());
  // Offset(in bytes) of the real storage in the basePtr memory block.
  // NB: this offset MUST be in bytes instead of numel, since we use
  // (storage_handle, offset)
  //     as key in shared_cache(multiprocessing/reduction.py).
  //     Offset in numel cannot uniquely represent a storage.
  PyTuple_SET_ITEM(tuple.get(), 3, _offset_bytes.release());
  PyTuple_SET_ITEM(tuple.get(), 4, _ref_counter.release());
  PyTuple_SET_ITEM(tuple.get(), 5, _ref_counter_offset.release());
  PyTuple_SET_ITEM(tuple.get(), 6, _event_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 7, _event_sync_required.release());
  return tuple.release();
#else
  TORCH_CHECK(false, "CUDA is not available");
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_releaseIPCCounter(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
#ifdef USE_CUDA
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 2, "tuple of 2 items expected");
  PyObject* _ref_counter = PyTuple_GET_ITEM(args, 0);
  PyObject* _ref_counter_offset = PyTuple_GET_ITEM(args, 1);
  if (!(PyBytes_Check(_ref_counter) &&
        THPUtils_checkLong(_ref_counter_offset))) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_release_ipc_counter in CUDA mode",
        1,
        "(bytes _ref_counter, int _ref_counter_offset)");
    return nullptr;
  }
  std::string ref_counter_handle = PyBytes_AS_STRING(_ref_counter);
  ptrdiff_t ref_counter_offset =
      static_cast<ptrdiff_t>(THPUtils_unpackLong(_ref_counter_offset));
  // We don't want to break existing code, so resource deletion is best
  // effort basis. Exception expected if producer process terminated
  // before consumer released data.
  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
  try {
    auto sptr = at::RefcountedMapAllocator::makeDataPtr(
        ref_counter_handle.c_str(),
        flags,
        sizeof(int64_t) * torch::CUDA_IPC_REF_COUNTER_FILE_SIZE,
        nullptr);
    *(static_cast<int64_t*>(sptr.get()) + ref_counter_offset) -= 1;
  } catch (c10::Error&) {
    // Already warned inside of producer process
  }
  Py_RETURN_NONE;
#else
  TORCH_CHECK(false, "CUDA is not available");
#endif
  END_HANDLE_TH_ERRORS
}

#ifdef USE_CUDA
static std::string THPStorage_bytesAsHandleString(PyObject* handle) {
  HANDLE_TH_ERRORS
  char* buffer = nullptr;
  Py_ssize_t handle_size = 0;
  if (PyBytes_AsStringAndSize(handle, &buffer, &handle_size) == -1) {
    TORCH_CHECK(handle_size == CUDA_IPC_HANDLE_SIZE, "incorrect handle");
  }
  return std::string(buffer, handle_size);
  END_HANDLE_TH_ERRORS_RET("")
}
#endif

static PyObject* THPStorage_newSharedCuda(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
#ifdef USE_CUDA
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 8, "tuple of 8 items expected");
  PyObject* _device = PyTuple_GET_ITEM(args, 0);
  PyObject* _handle = PyTuple_GET_ITEM(args, 1);
  PyObject* _size_bytes = PyTuple_GET_ITEM(args, 2);
  PyObject* _offset_bytes = PyTuple_GET_ITEM(args, 3);
  PyObject* _ref_counter = PyTuple_GET_ITEM(args, 4);
  PyObject* _ref_counter_offset = PyTuple_GET_ITEM(args, 5);
  PyObject* _event_handle = PyTuple_GET_ITEM(args, 6);
  PyObject* _event_sync_required = PyTuple_GET_ITEM(args, 7);
  if (!(THPUtils_checkLong(_device) && THPUtils_checkLong(_size_bytes) &&
        PyBytes_Check(_handle) && PyBytes_Check(_ref_counter) &&
        PyBytes_Check(_event_handle) && THPUtils_checkLong(_offset_bytes) &&
        THPUtils_checkLong(_ref_counter_offset) &&
        PyBool_Check(_event_sync_required))) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_new_shared in CUDA mode",
        1,
        "(int device, bytes handle, int storage_size_bytes, int storage_offset_bytes, bytes _ref_counter, int _ref_counter_offset, bytes event_handle, bool event_sync_required)");
    return nullptr;
  }

  size_t storage_size = THPUtils_unpackUInt64(_size_bytes) / sizeof(uint8_t);
  ptrdiff_t storage_offset_bytes =
      static_cast<ptrdiff_t>(THPUtils_unpackLong(_offset_bytes));

  const auto device = c10::checked_convert<c10::DeviceIndex>(
      THPUtils_unpackLong(_device), "c10::DeviceIndex");
  at::cuda::CUDAGuard device_guard(device);

  if (PyObject_IsTrue(_event_sync_required)) {
    // Ensure that producer prepared all tensor's data
    std::string s_ipc_event_handle =
        THPStorage_bytesAsHandleString(_event_handle);
    if (s_ipc_event_handle.empty()) {
      return nullptr;
    }
    auto ipc_event_handle = reinterpret_cast<const cudaIpcEventHandle_t*>(
        s_ipc_event_handle.c_str());
    at::cuda::CUDAEvent event(device, ipc_event_handle);
    event.block(c10::cuda::getCurrentCUDAStream(device));
  }

  std::string s_handle = THPStorage_bytesAsHandleString(_handle);
  if (s_handle.empty()) {
    return nullptr;
  }
  std::shared_ptr<void> basePtr =
      c10::cuda::CUDACachingAllocator::getIpcDevPtr(s_handle);

  // Offset the basePtr to reconstruct the real storage
  // devPtr = basePtr + storage_offset
  void* devPtr = basePtr.get();
  devPtr = static_cast<char*>(devPtr) + storage_offset_bytes;

  std::string ref_counter_handle = PyBytes_AS_STRING(_ref_counter);
  ptrdiff_t ref_counter_offset =
      static_cast<ptrdiff_t>(THPUtils_unpackLong(_ref_counter_offset));

  struct IpcDeleterContext {
    std::string ref_counter_handle;
    ptrdiff_t ref_counter_offset{};
    c10::DeviceIndex device{-1};
    torch::CudaIPCReceivedData received_data;
  };

  auto ctx = std::make_unique<IpcDeleterContext>();
  ctx->ref_counter_handle = std::move(ref_counter_handle);
  ctx->ref_counter_offset = ref_counter_offset;
  ctx->device = device;
  ctx->received_data.shared_ptr_ = std::move(basePtr);

  auto cur_device = at::cuda::current_device();
  c10::DataPtr data_ptr(
      devPtr,
      ctx.release(),
      +[](void* ctx_) {
        std::unique_ptr<IpcDeleterContext> ctx(
            static_cast<IpcDeleterContext*>(ctx_));
        ctx->received_data.shared_ptr_.reset();

        // Sync default stream to make sure all operations related to the
        // storage is finished (otherwise another process may reuse memory and
        // corrupt data)

        // Ideally all shared memory reference counting could be replaced by
        // sending untriggered CUDA event from the producer to consumer and
        // using this event as the criteria of memory release. However, CUDA
        // (atm 10.1) does not support the creation of untriggered events and
        // performance impact of having thousands of shared events is unknown.

        // TODO: Instead of cudaStreamSynchronize it is possible to add Stream
        // Callback and release counter inside of it (need to check performance
        // impact)

        // TODO: this isn't needed since CUDACachingAllocator already
        // synchronizes on free.
        at::cuda::stream_synchronize(
            c10::cuda::getCurrentCUDAStream(ctx->device));

        // We don't want to break existing code, so resource deletion is best
        // effort basis. Exception expected if producer process terminated
        // before consumer released data.
        int flags =
            at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
        try {
          auto sptr = at::RefcountedMapAllocator::makeDataPtr(
              ctx->ref_counter_handle.c_str(),
              flags,
              sizeof(int64_t) * torch::CUDA_IPC_REF_COUNTER_FILE_SIZE,
              nullptr);
          *(static_cast<int64_t*>(sptr.get()) + ctx->ref_counter_offset) -= 1;
        } catch (c10::Error&) {
          // Already warned inside of producer process
        }
      },
      at::Device(at::DeviceType::CUDA, cur_device));

  auto base = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      storage_size,
      std::move(data_ptr),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  base->set_resizable(false);
  base->set_received_cuda(true);

  return THPStorage_NewWithStorage(THPStorageClass, std::move(base));
#else
  TORCH_CHECK(false, "CUDA is not available");
#endif
  END_HANDLE_TH_ERRORS
}

// Returns an object that holds a "weak" pointer to the c10::StorageImpl.  This
// pointer keeps the c10::StorageImpl struct live, but does not retain the data
// pointer.
//
// NB: This does NOT preserve object identity when you call it multiple times
static PyObject* THPStorage_weakRef(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  c10::StorageImpl* storage = THPStorage_Unpack(self).unsafeGetStorageImpl();
  return PyLong_FromVoidPtr(c10::raw::intrusive_ptr::make_weak(storage));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_newWithWeakPtr(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg), "_new_with_weak_ptr(): arg must be an 'int'");
  c10::StorageImpl* weak_storage =
      static_cast<c10::StorageImpl*>(PyLong_AsVoidPtr(arg));
  if (auto* storage = c10::raw::weak_intrusive_ptr::lock(weak_storage)) {
    return THPStorage_Wrap(
        c10::intrusive_ptr<c10::StorageImpl>::reclaim(storage));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_freeWeakRef(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (Py_IsNone(arg)) {
    Py_RETURN_NONE;
  }
  TORCH_CHECK(
      THPUtils_checkLong(arg), "_free_weak_ref(): arg must be an 'int'");
  c10::StorageImpl* weak_storage =
      static_cast<c10::StorageImpl*>(PyLong_AsVoidPtr(arg));
  c10::raw::weak_intrusive_ptr::decref(weak_storage);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_expired(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "_expired(): arg must be an 'int'");
  c10::StorageImpl* weak_storage =
      static_cast<c10::StorageImpl*>(PyLong_AsVoidPtr(arg));
  return PyBool_FromLong(
      c10::raw::weak_intrusive_ptr::use_count(weak_storage) == 0);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_sharedFd(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  at::MapAllocator* ctx = nullptr;
  const auto& storage = THPStorage_Unpack(self);
  if (storage.device_type() == at::kCPU) {
    ctx = at::MapAllocator::fromDataPtr(storage.data_ptr());
  }

  TORCH_CHECK(ctx, "couldn't retrieve a shared file descriptor");
  return THPUtils_packInt32(ctx->fd());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_isShared(PyObject* self, PyObject* noargs) {
  const auto& storage = THPStorage_Unpack(self);
  if (storage.device_type() != at::kCPU && storage.device_type() != at::kMeta) {
    Py_RETURN_TRUE;
  }
  if (at::MapAllocator::fromDataPtr(storage.data_ptr()) ||
      THManagedMapAllocator::fromDataPtr(storage.data_ptr())) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

#ifdef USE_XPU
namespace {

inline constexpr int64_t XPU_IPC_REF_COUNTER_FILE_SIZE = 10000;
inline constexpr int64_t XPU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;

struct XpuIPCRefCountersFile final {
  XpuIPCRefCountersFile(std::string handle, uint64_t size, at::DataPtr data_ptr)
      : size_(size), handle_(std::move(handle)), refcounted_shared_mem_(std::move(data_ptr)) {}

  uint64_t* counter_ptr() {
    return static_cast<uint64_t*>(refcounted_shared_mem_.get()) + next_offset_;
  }

  void set_counter(uint64_t value) {
    *counter_ptr() = value;
  }

  bool have_offsets() {
    return next_offset_ < size_;
  }

  bool offsets_in_use() {
    return used_slots_;
  }

  uint64_t get_offset() {
    return next_offset_;
  }

  void rotate_offset() {
    next_offset_++;
    used_slots_++;
  }

  void return_offset(uint64_t offset) {
    (void)offset;
    used_slots_--;
  }

  const std::string& handle() {
    return handle_;
  }

 private:
  uint64_t next_offset_{0};
  uint64_t size_;
  uint64_t used_slots_{0};
  std::string handle_;
  at::DataPtr refcounted_shared_mem_;
};

class XpuIPCSentData final {
 public:
  XpuIPCSentData(
      std::string handle,
      uint64_t offset,
      uint64_t* counter_ptr,
      at::Device device)
      : handle_(std::move(handle)),
        offset_(offset),
        counter_ptr_(counter_ptr),
        device_(device) {}

  ~XpuIPCSentData();

  uint64_t counter_value() {
    return *counter_ptr_;
  }

  const std::string& handle() {
    return handle_;
  }

  uint64_t offset() {
    return offset_;
  }

  void set_original_ptr(at::DataPtr data_ptr) {
    original_ptr_ = std::move(data_ptr);
  }

 private:
  std::string handle_;
  uint64_t offset_;
  uint64_t* counter_ptr_;
  at::DataPtr original_ptr_;
  at::Device device_;
};

struct XpuIPCSentDataLimbo final {
  bool collect() {
    bool freed_memory = false;
    std::vector<std::unique_ptr<XpuIPCSentData>> kept_blocks;
    {
      std::lock_guard<std::mutex> lock(limbo_mutex_);
      kept_blocks.reserve(shared_blocks_.size());
      for (auto& sd : shared_blocks_) {
        if (sd->counter_value() > 0) {
          kept_blocks.push_back(std::move(sd));
        } else {
          freed_memory = true;
        }
      }
      shared_blocks_ = std::move(kept_blocks);
    }
    return freed_memory;
  }

  void add(std::unique_ptr<XpuIPCSentData> shared_block) {
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    shared_blocks_.push_back(std::move(shared_block));
    if (shared_blocks_.size() > XPU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO) {
      TORCH_WARN_ONCE(
          "XPU IPC tensors waiting on refcount release exceeded ",
          XPU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO,
          ". Consider ensuring consumers release shared tensors promptly.");
    }
  }

  uint64_t size() {
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    return shared_blocks_.size();
  }

 private:
  std::vector<std::unique_ptr<XpuIPCSentData>> shared_blocks_;
  std::mutex limbo_mutex_;
};

struct XpuIPCGlobalEntities final {
  XpuIPCGlobalEntities() = default;

  ~XpuIPCGlobalEntities() {
    alive = false;
  }

  void safe_clean_current_file() {
    std::lock_guard<std::mutex> lock(ref_counters_mutex_);
    if (next_available_ref_counters_file_ &&
        next_available_ref_counters_file_->offsets_in_use() == 0) {
      ref_counters_files_.erase(next_available_ref_counters_file_->handle());
      next_available_ref_counters_file_.reset();
    }
  }

  static bool alive;
  std::mutex ref_counters_mutex_;
  std::unordered_map<std::string, std::shared_ptr<XpuIPCRefCountersFile>>
      ref_counters_files_;
  std::shared_ptr<XpuIPCRefCountersFile> next_available_ref_counters_file_;
  XpuIPCSentDataLimbo limbo_;
};

bool XpuIPCGlobalEntities::alive = true;
XpuIPCGlobalEntities xpu_ipc_global_entities;

void ReturnXpuRefCounter(const std::string& handle, uint64_t offset) {
  if (!XpuIPCGlobalEntities::alive) {
    return;
  }
  std::lock_guard<std::mutex> lock(xpu_ipc_global_entities.ref_counters_mutex_);
  auto& map = xpu_ipc_global_entities.ref_counters_files_;
  auto it = map.find(handle);
  if (it != map.end()) {
    it->second->return_offset(offset);
    if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
      map.erase(handle);
    }
  }
}

XpuIPCSentData::~XpuIPCSentData() {
  if (!XpuIPCGlobalEntities::alive) {
    original_ptr_.release_context();
  }
  ReturnXpuRefCounter(handle_, offset_);
}

void XpuIPCSentDataDelete(void* ptr) {
  std::unique_ptr<XpuIPCSentData> sent_data(static_cast<XpuIPCSentData*>(ptr));
  if (!XpuIPCGlobalEntities::alive) {
    return;
  }
  if (sent_data->counter_value() > 0) {
    xpu_ipc_global_entities.limbo_.add(std::move(sent_data));
  }
  xpu_ipc_global_entities.limbo_.collect();
}

at::DataPtr GetNewRefCountedXpuSentData(void* data, at::Device device) {
  {
    std::lock_guard<std::mutex> lock(xpu_ipc_global_entities.ref_counters_mutex_);
    if (!xpu_ipc_global_entities.next_available_ref_counters_file_) {
      std::string ref_counter_handle = at::NewProcessWideShmHandle();
      int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
      at::DataPtr sptr = at::RefcountedMapAllocator::makeDataPtr(
          ref_counter_handle.c_str(),
          flags,
          sizeof(int64_t) * XPU_IPC_REF_COUNTER_FILE_SIZE,
          nullptr);
      auto rc = std::make_shared<XpuIPCRefCountersFile>(
          ref_counter_handle,
          XPU_IPC_REF_COUNTER_FILE_SIZE,
          std::move(sptr));
      xpu_ipc_global_entities.ref_counters_files_[ref_counter_handle] = rc;
      xpu_ipc_global_entities.next_available_ref_counters_file_ = rc;
    }
  }

  xpu_ipc_global_entities.next_available_ref_counters_file_->set_counter(1);
  auto sent_data = new XpuIPCSentData(
      xpu_ipc_global_entities.next_available_ref_counters_file_->handle(),
      xpu_ipc_global_entities.next_available_ref_counters_file_->get_offset(),
      xpu_ipc_global_entities.next_available_ref_counters_file_->counter_ptr(),
      device);

  xpu_ipc_global_entities.next_available_ref_counters_file_->rotate_offset();
  if (!xpu_ipc_global_entities.next_available_ref_counters_file_->have_offsets()) {
    xpu_ipc_global_entities.next_available_ref_counters_file_.reset();
  }
  return at::DataPtr(data, sent_data, XpuIPCSentDataDelete, device);
}

bool XpuIPCCollect() {
  if (!XpuIPCGlobalEntities::alive) {
    return true;
  }
  bool freed_memory = xpu_ipc_global_entities.limbo_.collect();
  if (xpu_ipc_global_entities.limbo_.size() == 0) {
    xpu_ipc_global_entities.safe_clean_current_file();
  }
  return freed_memory;
}

void ReleaseXpuIPCRefCounter(const std::string& handle, ptrdiff_t offset) {
  if (handle.empty()) {
    return;
  }
  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
  try {
    auto sptr = at::RefcountedMapAllocator::makeDataPtr(
        handle.c_str(),
        flags,
        sizeof(int64_t) * XPU_IPC_REF_COUNTER_FILE_SIZE,
        nullptr);
    *(static_cast<int64_t*>(sptr.get()) + offset) -= 1;
  } catch (c10::Error&) {
  }
  XpuIPCCollect();
}

struct XpuSharedStorageArgs {
  c10::DeviceIndex device;
  std::string handle;
  std::string event;
  std::string ref_counter_handle;
  size_t storage_size;
  ptrdiff_t storage_offset_bytes;
  ptrdiff_t ref_counter_offset;
};

class XpuIpcEvent {
 public:
  static XpuIpcEvent create(c10::DeviceIndex device) {
    return XpuIpcEvent(device, false, std::nullopt);
  }

  static XpuIpcEvent open(
      c10::DeviceIndex device,
      const std::string& ipc_pool_handle) {
    return XpuIpcEvent(device, true, ipc_pool_handle);
  }

  XpuIpcEvent(const XpuIpcEvent&) = delete;
  XpuIpcEvent& operator=(const XpuIpcEvent&) = delete;
  XpuIpcEvent(XpuIpcEvent&& other) noexcept
      : pool_(other.pool_),
        event_(other.event_),
        opened_ipc_pool_(other.opened_ipc_pool_) {
    other.release();
  }

  XpuIpcEvent& operator=(XpuIpcEvent&& other) noexcept {
    if (this != &other) {
      cleanup();
      pool_ = other.pool_;
      event_ = other.event_;
      opened_ipc_pool_ = other.opened_ipc_pool_;
      other.release();
    }
    return *this;
  }

  ~XpuIpcEvent() { cleanup(); }

  std::string exportHandle() const {
#ifndef _WIN32
    ze_ipc_event_pool_handle_t ipc_handle{};
    const auto& ze = at::detail::getXPUHooks().level_zero();
    TORCH_CHECK(
        pool_,
        "XPU IPC event pool is not initialized before export");
    TORCH_CHECK(
        ze.zeEventPoolGetIpcHandle(pool_, &ipc_handle) == ZE_RESULT_SUCCESS,
        "Failed to export XPU IPC event pool handle");
    return std::string(
        reinterpret_cast<const char*>(&ipc_handle), sizeof(ipc_handle));
#else
    return {};
#endif
  }

  void signal() const {
#ifndef _WIN32
    const auto& ze = at::detail::getXPUHooks().level_zero();
    TORCH_CHECK(event_, "XPU IPC event is not initialized");
    TORCH_CHECK(
        ze.zeEventHostSignal(event_) == ZE_RESULT_SUCCESS,
        "Failed to signal XPU IPC event");
#endif
  }

  void waitOnStream(const c10::xpu::XPUStream& stream) const {
#ifndef _WIN32
    TORCH_CHECK(event_, "XPU IPC event is not initialized");
    auto backend_event = sycl::backend_input_t<
        sycl::backend::ext_oneapi_level_zero,
        sycl::event>{
        event_, sycl::ext::oneapi::level_zero::ownership::keep};
    auto sycl_event = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
        backend_event, c10::xpu::get_device_context());
    std::vector<sycl::event> event_list{sycl_event};
    stream.queue().ext_oneapi_submit_barrier(event_list);
#else
    (void)stream;
#endif
  }

 private:
  void cleanup() {
#ifndef _WIN32
    const auto& ze = at::detail::getXPUHooks().level_zero();
    if (event_) {
      ze.zeEventDestroy(event_);
    }
    if (pool_) {
      if (opened_ipc_pool_) {
        ze.zeEventPoolCloseIpcHandle(pool_);
      } else {
        ze.zeEventPoolDestroy(pool_);
      }
    }
#endif
  }

  void release() noexcept {
    pool_ = nullptr;
    event_ = nullptr;
    opened_ipc_pool_ = false;
  }

  XpuIpcEvent(
      c10::DeviceIndex device,
      bool open_from_ipc,
      std::optional<std::string> ipc_pool_handle) {
#ifndef _WIN32
    const auto& ze = at::detail::getXPUHooks().level_zero();
    auto& sycl_device = c10::xpu::get_raw_device(device);
    auto& sycl_context = c10::xpu::get_device_context();
    auto l0_device =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
    auto l0_context =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);

    if (open_from_ipc) {
      TORCH_CHECK(ipc_pool_handle.has_value(), "Missing XPU IPC pool handle");
      TORCH_CHECK(
          ipc_pool_handle->size() == sizeof(ze_ipc_event_pool_handle_t),
          "Invalid XPU IPC event pool handle size");
      ze_ipc_event_pool_handle_t ipc_handle{};
      std::memcpy(
          &ipc_handle,
          ipc_pool_handle->data(),
          sizeof(ze_ipc_event_pool_handle_t));
      TORCH_CHECK(
          ze.zeEventPoolOpenIpcHandle(l0_context, ipc_handle, &pool_) ==
              ZE_RESULT_SUCCESS,
          "Failed to open XPU IPC event pool handle");
      opened_ipc_pool_ = true;
    } else {
      ze_event_pool_desc_t pool_desc{};
      pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
      pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_IPC;
      pool_desc.count = 1;
      TORCH_CHECK(
          ze.zeEventPoolCreate(l0_context, &pool_desc, 1, &l0_device, &pool_) ==
              ZE_RESULT_SUCCESS,
          "Failed to create XPU IPC event pool");
    }

    ze_event_desc_t event_desc{};
    event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    event_desc.index = 0;
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    TORCH_CHECK(
        ze.zeEventCreate(pool_, &event_desc, &event_) == ZE_RESULT_SUCCESS,
        "Failed to create XPU IPC event");
#else
    (void)device;
    (void)open_from_ipc;
    (void)ipc_pool_handle;
#endif
  }

  ze_event_pool_handle_t pool_{nullptr};
  ze_event_handle_t event_{nullptr};
  bool opened_ipc_pool_{false};
};

// Wrapper for XPU IPC event lifetime tracking (analogous to CUDA CudaIPCSentData).
// Represents a ref-guarded event that ensures producer-consumer sync.
struct XpuIpcEventRefGuard {
  std::shared_ptr<XpuIpcEvent> event_sp;

  XpuIpcEventRefGuard() = default;
  explicit XpuIpcEventRefGuard(XpuIpcEvent&& event)
      : event_sp(std::make_shared<XpuIpcEvent>(std::move(event))) {}

  bool has_event() const { return event_sp != nullptr; }
  void wait_on_stream(const c10::xpu::XPUStream& stream) const {
    if (has_event()) {
      event_sp->waitOnStream(stream);
    }
  }
};

bool isImportedStorage(const c10::StorageImpl& storage) {
  return const_cast<c10::StorageImpl&>(storage).received_cuda();
}

void markImportedStorage(c10::StorageImpl& storage) {
  storage.set_received_cuda(true);
}

THPObjectPtr createXpuShareTuple(const at::Storage& storage) {
  THPObjectPtr tuple(PyTuple_New(7));
  THPObjectPtr device(THPUtils_packInt32(storage.device().index()));
  THPObjectPtr handle(Py_NewRef(Py_None));
  THPObjectPtr event(PyBytes_FromStringAndSize(nullptr, 0));
  THPObjectPtr ref_counter(Py_NewRef(Py_None));
  THPObjectPtr ref_counter_offset(THPUtils_packInt32(0));
  THPObjectPtr size_bytes(THPUtils_packUInt64(storage.nbytes()));
  THPObjectPtr offset_bytes(THPUtils_packInt32(0));

  if (storage.data()) {
    c10::xpu::syncStreamsOnDevice(storage.device().index());
    auto shandle =
        c10::xpu::XPUCachingAllocator::shareIpcHandle(storage.mutable_data());
    auto ipc_event = XpuIpcEvent::create(storage.device().index());
    ipc_event.signal();
    handle = PyBytes_FromStringAndSize(
        shandle.handle.c_str(), static_cast<Py_ssize_t>(shandle.handle.size()));
    const auto event_handle = ipc_event.exportHandle();
    event = PyBytes_FromStringAndSize(
        event_handle.c_str(), static_cast<Py_ssize_t>(event_handle.size()));
    offset_bytes = PyLong_FromSsize_t(static_cast<Py_ssize_t>(shandle.offset));

    at::DataPtr sent_data_ptr =
        GetNewRefCountedXpuSentData(storage.mutable_data(), storage.device());
    auto old_data_ptr = storage.set_data_ptr(std::move(sent_data_ptr));
    auto sent_data =
        static_cast<XpuIPCSentData*>(storage.data_ptr().get_context());
    sent_data->set_original_ptr(std::move(old_data_ptr));
    ref_counter = PyBytes_FromString(sent_data->handle().c_str());
    ref_counter_offset = THPUtils_packUInt64(sent_data->offset());
  }

  if (!tuple || !device || !handle || !event || !size_bytes || !offset_bytes ||
      !ref_counter || !ref_counter_offset) {
    return {};
  }

  PyTuple_SET_ITEM(tuple.get(), 0, device.release());
  PyTuple_SET_ITEM(tuple.get(), 1, handle.release());
  PyTuple_SET_ITEM(tuple.get(), 2, event.release());
  PyTuple_SET_ITEM(tuple.get(), 3, ref_counter.release());
  PyTuple_SET_ITEM(tuple.get(), 4, ref_counter_offset.release());
  PyTuple_SET_ITEM(tuple.get(), 5, size_bytes.release());
  PyTuple_SET_ITEM(tuple.get(), 6, offset_bytes.release());
  return tuple;
}

bool parseXpuSharedStorageArgs(PyObject* args, XpuSharedStorageArgs& parsed) {
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 7, "tuple of 7 items expected");
  PyObject* device = PyTuple_GET_ITEM(args, 0);
  PyObject* handle = PyTuple_GET_ITEM(args, 1);
  PyObject* event = PyTuple_GET_ITEM(args, 2);
  PyObject* ref_counter = PyTuple_GET_ITEM(args, 3);
  PyObject* ref_counter_offset = PyTuple_GET_ITEM(args, 4);
  PyObject* size_bytes = PyTuple_GET_ITEM(args, 5);
  PyObject* offset_bytes = PyTuple_GET_ITEM(args, 6);

  if (!(THPUtils_checkLong(device) && PyBytes_Check(handle) &&
        PyBytes_Check(event) && PyBytes_Check(ref_counter) &&
        THPUtils_checkLong(ref_counter_offset) && THPUtils_checkLong(size_bytes) &&
        THPUtils_checkLong(offset_bytes))) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_new_shared in XPU mode",
        1,
        "(int device, bytes handle, bytes event, bytes ref_counter, int ref_counter_offset, int storage_size_bytes, int storage_offset_bytes)");
    return false;
  }

  parsed.storage_size = THPUtils_unpackUInt64(size_bytes) / sizeof(uint8_t);
  parsed.storage_offset_bytes =
      static_cast<ptrdiff_t>(THPUtils_unpackLong(offset_bytes));
  parsed.device = c10::checked_convert<c10::DeviceIndex>(
      THPUtils_unpackLong(device), "c10::DeviceIndex");

  char* handle_data = nullptr;
  Py_ssize_t handle_size = 0;
  if (PyBytes_AsStringAndSize(handle, &handle_data, &handle_size) == -1) {
    TORCH_CHECK(false, "Failed to extract IPC handle bytes");
  }
  parsed.handle = std::string(handle_data, handle_size);

  char* event_data = nullptr;
  Py_ssize_t event_size = 0;
  if (PyBytes_AsStringAndSize(event, &event_data, &event_size) == -1) {
    TORCH_CHECK(false, "Failed to extract IPC event bytes");
  }
  parsed.event = std::string(event_data, event_size);

  char* ref_counter_data = nullptr;
  Py_ssize_t ref_counter_size = 0;
  if (PyBytes_AsStringAndSize(
          ref_counter, &ref_counter_data, &ref_counter_size) == -1) {
    TORCH_CHECK(false, "Failed to extract IPC ref-counter bytes");
  }
  parsed.ref_counter_handle = std::string(ref_counter_data, ref_counter_size);
  parsed.ref_counter_offset =
      static_cast<ptrdiff_t>(THPUtils_unpackLong(ref_counter_offset));
  return true;
}

c10::intrusive_ptr<at::StorageImpl> createStorageImplFromXpuShared(
    const XpuSharedStorageArgs& args) {
  c10::DeviceGuard device_guard(c10::Device(c10::kXPU, args.device));
  XpuIpcEventRefGuard event_guard;
  if (!args.event.empty()) {
    XpuIpcEvent event = XpuIpcEvent::open(args.device, args.event);
    event_guard = XpuIpcEventRefGuard(std::move(event));
    event_guard.wait_on_stream(c10::xpu::getCurrentXPUStream(args.device));
  }
  auto base_ptr =
      c10::xpu::XPUCachingAllocator::getIpcDevPtr(args.handle, args.device);

  struct XpuIpcDeleterContext {
    std::shared_ptr<void> base_ptr;
    XpuIpcEventRefGuard event_guard;
    std::string ref_counter_handle;
    ptrdiff_t ref_counter_offset{0};
    c10::DeviceIndex device{-1};
  };

  auto ctx = std::make_unique<XpuIpcDeleterContext>();
  ctx->base_ptr = std::move(base_ptr);
  ctx->event_guard = std::move(event_guard);
  ctx->ref_counter_handle = args.ref_counter_handle;
  ctx->ref_counter_offset = args.ref_counter_offset;
  ctx->device = args.device;

  void* dev_ptr = ctx->base_ptr.get();
  dev_ptr = static_cast<char*>(dev_ptr) + args.storage_offset_bytes;

  c10::DataPtr data_ptr(
      dev_ptr,
      ctx.release(),
      +[](void* ctx_) {
        std::unique_ptr<XpuIpcDeleterContext> ctx(
            static_cast<XpuIpcDeleterContext*>(ctx_));
        if (ctx->device >= 0) {
          c10::xpu::syncStreamsOnDevice(ctx->device);
        }
        ReleaseXpuIPCRefCounter(
            ctx->ref_counter_handle, ctx->ref_counter_offset);
        ctx->base_ptr.reset();
      },
      at::Device(at::DeviceType::XPU, args.device));

  auto storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      args.storage_size,
      std::move(data_ptr),
      nullptr,
      false);
  markImportedStorage(*storage);
  return storage;
}

} // namespace

static PyObject* THPStorage_shareXpu(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(self);
  TORCH_CHECK(
      storage.device_type() == at::kXPU, "_share_xpu_: only available on XPU");

  c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();
  if (isImportedStorage(*storage_impl)) {
    TORCH_CHECK(
        false,
        "Attempted to send XPU tensor received from another process; "
        "this is not currently supported. Consider cloning before sending.");
  }

  at::DeviceGuard device_guard(storage.device());

  try {
    auto tuple = createXpuShareTuple(storage);
    return tuple.release();
  } catch (const c10::Error& e) {
    TORCH_CHECK(false, "Failed to get XPU IPC handle: ", e.what());
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_releaseIPCCounterXpu(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 2, "tuple of 2 items expected");
  PyObject* ref_counter = PyTuple_GET_ITEM(args, 0);
  PyObject* ref_counter_offset = PyTuple_GET_ITEM(args, 1);
  if (!(PyBytes_Check(ref_counter) && THPUtils_checkLong(ref_counter_offset))) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_release_ipc_counter_xpu in XPU mode",
        1,
        "(bytes ref_counter, int ref_counter_offset)");
    return nullptr;
  }

  std::string ref_counter_handle = PyBytes_AS_STRING(ref_counter);
  ptrdiff_t offset =
      static_cast<ptrdiff_t>(THPUtils_unpackLong(ref_counter_offset));
  ReleaseXpuIPCRefCounter(ref_counter_handle, offset);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_newSharedXpu(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  XpuSharedStorageArgs parsed;
  if (!parseXpuSharedStorageArgs(args, parsed)) {
    return nullptr;
  }

  try {
    auto storage = createStorageImplFromXpuShared(parsed);
    return THPStorage_NewWithStorage(THPStorageClass, std::move(storage));
  } catch (const c10::Error& e) {
    TORCH_CHECK(false, "Failed to open XPU IPC memory: ", e.what());
  }
  END_HANDLE_TH_ERRORS
}

#endif // USE_XPU

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef THPStorage_sharingMethods[] = {
    {"_new_with_weak_ptr",
     THPStorage_newWithWeakPtr,
     METH_O | METH_CLASS,
     nullptr},
    {"_share_cuda_", THPStorage_shareCuda, METH_NOARGS, nullptr},
    {"_new_shared_cuda",
     THPStorage_newSharedCuda,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_release_ipc_counter_cuda",
     THPStorage_releaseIPCCounter,
     METH_VARARGS | METH_STATIC,
     nullptr},
#ifdef USE_XPU
    {"_share_xpu_", THPStorage_shareXpu, METH_NOARGS, nullptr},
    {"_new_shared_xpu",
     THPStorage_newSharedXpu,
     METH_VARARGS | METH_STATIC,
     nullptr},
  {"_release_ipc_counter_xpu",
   THPStorage_releaseIPCCounterXpu,
   METH_VARARGS | METH_STATIC,
   nullptr},
#endif
    {"_share_fd_cpu_", THPStorage_shareFd, METH_NOARGS, nullptr},
    {"_new_shared_fd_cpu",
     THPStorage_newSharedFd,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_new_using_fd_cpu",
     THPStorage_pyNewFdStorage,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_share_filename_cpu_", THPStorage_shareFilename, METH_NOARGS, nullptr},
    {"_new_shared_filename_cpu",
     THPStorage_newSharedFilename,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_new_using_filename_cpu",
     THPStorage_pyNewFilenameStorage,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_weak_ref", THPStorage_weakRef, METH_NOARGS, nullptr},
    {"_free_weak_ref", THPStorage_freeWeakRef, METH_O | METH_STATIC, nullptr},
    {"_expired", THPStorage_expired, METH_O | METH_STATIC, nullptr},
    {"_shared_decref", THPStorage_sharedDecref, METH_NOARGS, nullptr},
    {"_shared_incref", THPStorage_sharedIncref, METH_NOARGS, nullptr},
    {"_get_shared_fd", THPStorage_sharedFd, METH_NOARGS, nullptr},
    {"is_shared", THPStorage_isShared, METH_NOARGS, nullptr},
    {nullptr}};

PyMethodDef* THPStorage_getSharingMethods() {
  return THPStorage_sharingMethods;
}
