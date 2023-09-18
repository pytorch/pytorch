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
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <ATen/MapAllocator.h>
#include <ATen/StorageUtils.h>
#include <torch/csrc/utils/python_numbers.h>
#include <atomic>
#include <string>

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
  Py_INCREF(self);
  return self;
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
          /*resizable=*/false),
      c10::impl::PyInterpreterStatus::TAGGED_BY_US);
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
  THPUtils_assert(PyTuple_GET_SIZE(args) == 3, "tuple of 3 items expected");
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
          /*resizable=*/false),
      c10::impl::PyInterpreterStatus::TAGGED_BY_US);
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
      THPStorageClass,
      at::new_shm_fd_storage(size),
      c10::impl::PyInterpreterStatus::TAGGED_BY_US);
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
  THPUtils_assert(PyTuple_GET_SIZE(args) == 2, "tuple of 2 items expected");
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
  int tmp_fd = (int)THPUtils_unpackLong(_tmp_fd);
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
          /*resizable=*/false),
      c10::impl::PyInterpreterStatus::TAGGED_BY_US);
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
    AT_ERROR(
        "Attempted to send CUDA tensor received from another process; this is not currently supported. Consider cloning before sending.");
  }

  at::DeviceGuard device_guard(storage.device());
  THPObjectPtr tuple(PyTuple_New(8));
  THPObjectPtr device(THPUtils_packInt32(storage.device().index()));
  THPObjectPtr _handle(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr size_bytes(THPUtils_packUInt64(storage.nbytes()));
  THPObjectPtr _offset_bytes(THPUtils_packInt32(0));
  THPObjectPtr _ref_counter(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr _ref_counter_offset(THPUtils_packInt32(0));
  THPObjectPtr _event_handle(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr _event_sync_required(Py_None);
  Py_INCREF(Py_None);
  if (storage.data()) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t base_size;
    void* base_ptr = c10::cuda::CUDACachingAllocator::getBaseAllocation(
        storage.mutable_data(), &base_size);
    ptrdiff_t offset_bytes = (char*)storage.data() - (char*)base_ptr;

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    cudaIpcMemHandle_t handle;
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, base_ptr));

    _handle = PyBytes_FromStringAndSize((char*)&handle, CUDA_IPC_HANDLE_SIZE);
    _offset_bytes = PyLong_FromSsize_t((Py_ssize_t)offset_bytes);

    // Put Storage Data behind new ref counting context
    // See Note [CUDA IPC Refcounting implementation explained]
    at::DataPtr sent_data_ptr = torch::GetNewRefCountedSentData(
        storage.mutable_data(), storage.device());
    auto old_data_ptr = storage.set_data_ptr(std::move(sent_data_ptr));
    auto sent_data =
        static_cast<torch::CudaIPCSentData*>(storage.data_ptr().get_context());
    sent_data->set_original_ptr(std::move(old_data_ptr));
    _ref_counter = PyBytes_FromString((sent_data->handle()).c_str());
    _ref_counter_offset = THPUtils_packInt64(sent_data->offset());

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    cudaIpcEventHandle_t ipc_event_handle;

    if (sent_data->event_sync_required_) {
      C10_CUDA_CHECK(
          cudaIpcGetEventHandle(&ipc_event_handle, sent_data->event_));
    }

    _event_handle = PyBytes_FromStringAndSize(
        (char*)&ipc_event_handle, CUDA_IPC_HANDLE_SIZE);
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
  THPUtils_assert(PyTuple_GET_SIZE(args) == 2, "tuple of 2 items expected");
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
      (ptrdiff_t)THPUtils_unpackLong(_ref_counter_offset);
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
  } catch (c10::Error& err) {
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
  char* buffer = nullptr;
  Py_ssize_t handle_size = 0;
  if (PyBytes_AsStringAndSize(handle, &buffer, &handle_size) == -1) {
    THPUtils_assertRet(
        "", handle_size == CUDA_IPC_HANDLE_SIZE, "incorrect handle");
  }
  THPUtils_assertRet(
      "", handle_size == CUDA_IPC_HANDLE_SIZE, "incorrect handle size");
  return std::string(buffer, handle_size);
}
#endif

static PyObject* THPStorage_newSharedCuda(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
#ifdef USE_CUDA
  THPUtils_assert(PyTuple_GET_SIZE(args) == 8, "tuple of 8 items expected");
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

  size_t storage_size =
      (size_t)THPUtils_unpackLong(_size_bytes) / sizeof(uint8_t);
  ptrdiff_t storage_offset_bytes =
      (ptrdiff_t)THPUtils_unpackLong(_offset_bytes);

  int64_t device = THPUtils_unpackLong(_device);
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
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    cudaEvent_t event;
    cudaIpcOpenEventHandle(&event, *ipc_event_handle);
    C10_CUDA_CHECK(
        cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream(device), event, 0));
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
  devPtr = (char*)devPtr + storage_offset_bytes;

  std::string ref_counter_handle = PyBytes_AS_STRING(_ref_counter);
  ptrdiff_t ref_counter_offset =
      (ptrdiff_t)THPUtils_unpackLong(_ref_counter_offset);

  struct IpcDeleterContext {
    std::string ref_counter_handle;
    ptrdiff_t ref_counter_offset;
    int64_t device;
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
        } catch (c10::Error& err) {
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

  return THPStorage_NewWithStorage(
      THPStorageClass,
      std::move(base),
      c10::impl::PyInterpreterStatus::TAGGED_BY_US);
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

PyObject* THPStorage_newWithWeakPtr(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "_new_with_weak_ptr(): arg must be an 'int'");
  c10::StorageImpl* weak_storage = (c10::StorageImpl*)PyLong_AsVoidPtr(arg);
  if (auto* storage = c10::raw::weak_intrusive_ptr::lock(weak_storage)) {
    return THPStorage_Wrap(
        c10::intrusive_ptr<c10::StorageImpl>::reclaim(storage));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPStorage_freeWeakRef(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (arg == Py_None) {
    Py_RETURN_NONE;
  }
  THPUtils_assert(
      THPUtils_checkLong(arg), "_free_weak_ref(): arg must be an 'int'");
  c10::StorageImpl* weak_storage = (c10::StorageImpl*)PyLong_AsVoidPtr(arg);
  c10::raw::weak_intrusive_ptr::decref(weak_storage);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPStorage_expired(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "_expired(): arg must be an 'int'");
  c10::StorageImpl* weak_storage = (c10::StorageImpl*)PyLong_AsVoidPtr(arg);
  return PyBool_FromLong(
      c10::raw::weak_intrusive_ptr::use_count(weak_storage) == 0);
  END_HANDLE_TH_ERRORS
}

PyObject* THPStorage_sharedFd(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  at::MapAllocator* ctx = nullptr;
  const auto& storage = THPStorage_Unpack(self);
  if (storage.device_type() == at::kCPU) {
    ctx = at::MapAllocator::fromDataPtr(storage.data_ptr());
  }

  THPUtils_assert(ctx, "couldn't retrieve a shared file descriptor");
  return THPUtils_packInt32(ctx->fd());
  END_HANDLE_TH_ERRORS
}

PyObject* THPStorage_isShared(PyObject* self, PyObject* noargs) {
  const auto& storage = THPStorage_Unpack(self);
  if (storage.device_type() == at::kCUDA) {
    Py_RETURN_TRUE;
  }
  if (at::MapAllocator::fromDataPtr(storage.data_ptr()) ||
      THManagedMapAllocator::fromDataPtr(storage.data_ptr())) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

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
