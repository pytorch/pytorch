#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include <torch/csrc/utils/python_numbers.h>
#include <random>

static PyObject * THPStorage_(sharedDecref)(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
#ifndef THC_GENERIC_FILE
  THWStorage *storage = self->cdata;
  THManagedMapAllocator *ctx = THManagedMapAllocator::fromDataPtr(storage->data_ptr());
  if (ctx) {
    ctx->decref();
  }
#endif
  Py_INCREF(self);
  return (PyObject *)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(sharedIncref)(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
#ifndef THC_GENERIC_FILE
  THWStorage *storage = self->cdata;
  THManagedMapAllocator *ctx = THManagedMapAllocator::fromDataPtr(storage->data_ptr());
  if (ctx) {
    ctx->incref();
  }
#endif
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

#ifndef THC_GENERIC_FILE
// TODO: move this somewhere - we only need one version
static std::string THPStorage_(__newHandle)() {
  static std::random_device rd;
  std::string handle = "/torch_";
#ifdef _MSC_VER
  handle += std::to_string(GetCurrentProcessId());
#else
  handle += std::to_string(getpid());
#endif
  handle += "_";
  handle += std::to_string(rd());
  return handle;
}

static THWStorage* THPStorage_(newFilenameStorage)(ptrdiff_t size)
{
  int flags = TH_ALLOCATOR_MAPPED_SHAREDMEM | TH_ALLOCATOR_MAPPED_EXCLUSIVE;
  std::string handle = THPStorage_(__newHandle)();
  return THWStorage_(newWithDataAndAllocator)(
      THManagedMapAllocator::makeDataPtr("", handle.c_str(), flags, size * sizeof(scalar_t)), size, /* allocator */ nullptr);
}

static PyObject * THPStorage_(pyNewFilenameStorage)(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  long long size;
  if (!PyArg_ParseTuple(args, "L", &size)) {
    return nullptr;
  }
  return THPStorage_(New)(THPStorage_(newFilenameStorage)(size));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(shareFilename)(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  THWStorage *storage = self->cdata;
  THManagedMapAllocator *ctx;
  // Storage is already in shared memory, just return a handle
  if ((ctx = THManagedMapAllocator::fromDataPtr(storage->data_ptr()))) {
    // done
  } else {
    // TODO: retry on collision
    // TODO: free GIL - but remember to reacquire it when an exception is thrown
    THWStoragePtr new_storage(
        THPStorage_(newFilenameStorage)(storage->nbytes() / sizeof(scalar_t)));
    THWStorage_(copy)(new_storage, storage);
    THWStorage_(swap)(storage, new_storage);
    ctx = THManagedMapAllocator::fromDataPtr(storage->data_ptr());
    AT_ASSERT(ctx);
  }

  THPObjectPtr manager_handle(PyBytes_FromString(ctx->manager_handle()));
  if (!manager_handle) return nullptr;
  THPObjectPtr storage_handle(PyBytes_FromString(ctx->filename()));
  if (!storage_handle) return nullptr;
  THPObjectPtr size(THPUtils_packUInt64(storage->nbytes() / sizeof(scalar_t)));
  if (!size) return nullptr;

  THPObjectPtr tuple(PyTuple_New(3));
  if (!tuple) return nullptr;
  PyTuple_SET_ITEM(tuple.get(), 0, manager_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 1, storage_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 2, size.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(newSharedFilename)(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 3, "tuple of 3 items expected");
  PyObject *_manager_handle = PyTuple_GET_ITEM(args, 0);
  PyObject *_object_handle = PyTuple_GET_ITEM(args, 1);
  PyObject *_size = PyTuple_GET_ITEM(args, 2);
  if (!PyBytes_Check(_manager_handle) || !PyBytes_Check(_object_handle) || !THPUtils_checkLong(_size)) {
    THPUtils_invalidArguments(args, nullptr, "_new_shared in file system mode", 1,
        "a handle (string/bytes) and storage size (int)");
    return nullptr;
  }
  const char *manager_handle = PyBytes_AS_STRING(_manager_handle);
  const char *object_handle = PyBytes_AS_STRING(_object_handle);
  int64_t size = THPUtils_unpackLong(_size);
  int flags = TH_ALLOCATOR_MAPPED_SHAREDMEM |
              TH_ALLOCATOR_MAPPED_NOCREATE;
  return THPStorage_(New)(
          THWStorage_(newWithDataAndAllocator)(
            THManagedMapAllocator::makeDataPtr(manager_handle, object_handle, flags, size * sizeof(scalar_t)),
            size,
            /* allocator */ nullptr));
  END_HANDLE_TH_ERRORS
}

static THWStorage* THPStorage_(newFdStorage)(ptrdiff_t size)
{
  int flags = TH_ALLOCATOR_MAPPED_SHAREDMEM |
              TH_ALLOCATOR_MAPPED_EXCLUSIVE |
              TH_ALLOCATOR_MAPPED_KEEPFD |
              TH_ALLOCATOR_MAPPED_UNLINK;
  std::string handle = THPStorage_(__newHandle)();
  auto sptr = THMapAllocator::makeDataPtr(handle.c_str(), flags, size * sizeof(scalar_t), nullptr);
  return THWStorage_(newWithDataAndAllocator)(std::move(sptr), size, /* allocator */ nullptr);
}

static PyObject * THPStorage_(pyNewFdStorage)(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  long long size;
  if (!PyArg_ParseTuple(args, "L", &size)) {
    return nullptr;
  }
  return THPStorage_(New)(THPStorage_(newFdStorage)(size));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(shareFd)(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  THWStorage *storage = self->cdata;
  THMapAllocator *ctx;
  // Storage is already in shared memory, just return a handle
  if ((ctx = THMapAllocator::fromDataPtr(storage->data_ptr()))) {
    // done
  } else {
    THWStoragePtr new_storage(
        THPStorage_(newFdStorage)(storage->nbytes() / sizeof(scalar_t)));
    THWStorage_(copy)(new_storage, storage);
    THWStorage_(swap)(storage, new_storage);
    ctx = THMapAllocator::fromDataPtr(storage->data_ptr());
    AT_ASSERT(ctx);
  }

  THPObjectPtr storage_handle(THPUtils_packInt32(ctx->fd()));
  if (!storage_handle) return nullptr;
  THPObjectPtr size(THPUtils_packUInt64(storage->nbytes() / sizeof(scalar_t)));
  if (!size) return nullptr;

  THPObjectPtr tuple(PyTuple_New(2));
  if (!tuple) return nullptr;
  PyTuple_SET_ITEM(tuple.get(), 0, storage_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 1, size.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(newSharedFd)(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 2, "tuple of 2 items expected");
  PyObject *_tmp_fd = PyTuple_GET_ITEM(args, 0);
  PyObject *_size = PyTuple_GET_ITEM(args, 1);
  if (!THPUtils_checkLong(_tmp_fd) || !THPUtils_checkLong(_size)) {
    THPUtils_invalidArguments(args, nullptr, "_new_shared in file descriptor mode",
        1, "a file descriptor (int) and storage size (int)");
    return nullptr;
  }
  int fd;
  int tmp_fd = (int) THPUtils_unpackLong(_tmp_fd);
  int64_t size = THPUtils_unpackLong(_size);
  if ((fd = dup(tmp_fd)) == -1) {
    THPUtils_setError("could not duplicate a shared memory file descriptor");
    return nullptr;
  }

  int flags = TH_ALLOCATOR_MAPPED_SHAREDMEM |
              TH_ALLOCATOR_MAPPED_NOCREATE |
              TH_ALLOCATOR_MAPPED_KEEPFD |
              TH_ALLOCATOR_MAPPED_FROMFD;
  return THPStorage_(New)(
          THWStorage_(newWithDataAndAllocator)(
            // TODO: Maybe we should read out the scalar_t size and use it for size
            THMapAllocator::makeDataPtr(WITH_FD, nullptr, fd, flags, size * sizeof(scalar_t), nullptr),
            size, /* allocator */ nullptr));
  END_HANDLE_TH_ERRORS
}

#else // THC_GENERIC_FILE

static PyObject * THPStorage_(shareCuda)(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  THWStorage *storage = self->cdata;

  if (storage->received_cuda()) {
    AT_ERROR(
        "Attempted to send CUDA tensor received from another process; this is not currently supported. Consider cloning before sending.");
  }

  at::DeviceGuard device_guard(storage->device());
  THPObjectPtr tuple(PyTuple_New(8));
  THPObjectPtr device(THPUtils_packInt32(storage->device().index()));
  THPObjectPtr _handle(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr size_bytes(THPUtils_packUInt64(storage->nbytes()));
  THPObjectPtr _offset_bytes(THPUtils_packInt32(0));
  THPObjectPtr _ref_counter(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr _ref_counter_offset(THPUtils_packInt32(0));
  THPObjectPtr _event_handle(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr _event_sync_required(Py_None);
  Py_INCREF(Py_None);
  if (THWStorage_(data)(LIBRARY_STATE storage)) {
    size_t base_size;
    void *base_ptr = c10::cuda::CUDACachingAllocator::getBaseAllocation(THWStorage_(data)(LIBRARY_STATE storage), &base_size);
    ptrdiff_t offset_bytes = (char*)storage->data<scalar_t>() - (char*)base_ptr;

    cudaIpcMemHandle_t handle;
    THCudaCheck(cudaIpcGetMemHandle(&handle, base_ptr));

    _handle = PyBytes_FromStringAndSize((char *)&handle, CUDA_IPC_HANDLE_SIZE);
    _offset_bytes = PyLong_FromSsize_t((Py_ssize_t)offset_bytes);

    // Put Storage Data behind new ref counting context
    // See Note [CUDA IPC Refcounting implementation explained]
    at::DataPtr sent_data_ptr = torch::GetNewRefCountedSentData(storage->data(), storage->device());
    auto old_data_ptr = storage->set_data_ptr(std::move(sent_data_ptr));
    auto sent_data  =  static_cast<torch::CudaIPCSentData*>(storage->data_ptr().get_context());
    sent_data->set_original_ptr(std::move(old_data_ptr));
    _ref_counter = PyBytes_FromString((sent_data->handle()).c_str());
    _ref_counter_offset = THPUtils_packInt64(sent_data->offset());


    cudaIpcEventHandle_t ipc_event_handle;

#ifndef __HIP_PLATFORM_HCC__
    if (sent_data->event_sync_required_) {
      THCudaCheck(cudaIpcGetEventHandle(&ipc_event_handle, sent_data->event_));
    }
#else
    // ipc_event_handle unused in storage receiver, we can leave it uninitialized.
#endif

    _event_handle = PyBytes_FromStringAndSize((char *)&ipc_event_handle, CUDA_IPC_HANDLE_SIZE);
    _event_sync_required = PyBool_FromLong(sent_data->event_sync_required_);

  }

  if (!tuple || !device || !_handle || !size_bytes || !_offset_bytes || !_event_handle) {
    return nullptr;
  }
  PyTuple_SET_ITEM(tuple.get(), 0, device.release());
  // cudaIpcMemHandle_t(of basePtr)
  PyTuple_SET_ITEM(tuple.get(), 1, _handle.release());
  // Size(in bytes) of the real storage, note this is not the size of basePtr memory block.
  PyTuple_SET_ITEM(tuple.get(), 2, size_bytes.release());
  // Offset(in bytes) of the real storage in the basePtr memory block.
  // NB: this offset MUST be in bytes instead of numel, since we use (storage_handle, offset)
  //     as key in shared_cache(multiprocessing/reduction.py).
  //     Offset in numel cannot uniquely represent a storage.
  PyTuple_SET_ITEM(tuple.get(), 3, _offset_bytes.release());
  PyTuple_SET_ITEM(tuple.get(), 4, _ref_counter.release());
  PyTuple_SET_ITEM(tuple.get(), 5, _ref_counter_offset.release());
  PyTuple_SET_ITEM(tuple.get(), 6, _event_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 7, _event_sync_required.release());
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(releaseIPCCounter)(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 2, "tuple of 2 items expected");
  PyObject *_ref_counter = PyTuple_GET_ITEM(args, 0);
  PyObject *_ref_counter_offset = PyTuple_GET_ITEM(args, 1);
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
  int flags =
      TH_ALLOCATOR_MAPPED_SHAREDMEM | TH_ALLOCATOR_MAPPED_NOCREATE;
  try {
    auto sptr = THRefcountedMapAllocator::makeDataPtr(
        ref_counter_handle.c_str(),
        flags,
        sizeof(int64_t) * torch::CUDA_IPC_REF_COUNTER_FILE_SIZE,
        nullptr);
    *(static_cast<int64_t*>(sptr.get()) + ref_counter_offset) -= 1;
  } catch (c10::Error& err) {
    // Already warned inside of producer process
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static std::string THPStorage_(bytesAsHandleString)(PyObject *handle) {
  char* buffer;
  Py_ssize_t handle_size;
  if (PyBytes_AsStringAndSize(handle, &buffer, &handle_size) == -1) {
    return nullptr;
  }
  THPUtils_assert(
      handle_size == CUDA_IPC_HANDLE_SIZE, "incorrect handle size");
  return std::string(buffer, handle_size);
}

static PyObject * THPStorage_(newSharedCuda)(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 8, "tuple of 8 items expected");
  PyObject *_device = PyTuple_GET_ITEM(args, 0);
  PyObject *_handle = PyTuple_GET_ITEM(args, 1);
  PyObject *_size_bytes = PyTuple_GET_ITEM(args, 2);
  PyObject *_offset_bytes = PyTuple_GET_ITEM(args, 3);
  PyObject *_ref_counter = PyTuple_GET_ITEM(args, 4);
  PyObject *_ref_counter_offset = PyTuple_GET_ITEM(args, 5);
  PyObject *_event_handle = PyTuple_GET_ITEM(args, 6);
  PyObject *_event_sync_required = PyTuple_GET_ITEM(args, 7);
  if (!(THPUtils_checkLong(_device) && THPUtils_checkLong(_size_bytes) &&
        PyBytes_Check(_handle) && PyBytes_Check(_ref_counter) &&
        PyBytes_Check(_event_handle) && THPUtils_checkLong(_offset_bytes) &&
        THPUtils_checkLong(_ref_counter_offset) && PyBool_Check(_event_sync_required))) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "_new_shared in CUDA mode",
        1,
        "(int device, bytes handle, int storage_size_bytes, int storage_offset_bytes, bytes _ref_counter, int _ref_counter_offset, bytes event_handle, bool event_sync_required)");
    return nullptr;
  }

  size_t storage_size = (size_t)THPUtils_unpackLong(_size_bytes) / sizeof(scalar_t);
  ptrdiff_t storage_offset_bytes = (ptrdiff_t)THPUtils_unpackLong(_offset_bytes);

  int64_t device = THPUtils_unpackLong(_device);
  at::cuda::CUDAGuard device_guard(device);

#ifndef __HIP_PLATFORM_HCC__
  if (PyObject_IsTrue(_event_sync_required)) {
    // Ensure that producer prepared all tensor's data
    std::string s_ipc_event_handle =
        THPStorage_(bytesAsHandleString)(_event_handle);
    auto ipc_event_handle = reinterpret_cast<const cudaIpcEventHandle_t*>(
        s_ipc_event_handle.c_str());
    cudaEvent_t event;
    cudaIpcOpenEventHandle(&event, *ipc_event_handle);
    AT_CUDA_CHECK(
        cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream(device), event, 0));
  }
#else
  // Already synchronized inside producer stream
#endif

  std::string s_handle = THPStorage_(bytesAsHandleString)(_handle);
  std::shared_ptr<void> basePtr = c10::cuda::CUDACachingAllocator::getIpcDevPtr(s_handle);

  // Offset the basePtr to reconstruct the real storage
  // devPtr = basePtr + storage_offset
  void* devPtr = basePtr.get();
  devPtr = (char*)devPtr + storage_offset_bytes;

  std::string ref_counter_handle = PyBytes_AS_STRING(_ref_counter);
  ptrdiff_t ref_counter_offset = (ptrdiff_t)THPUtils_unpackLong(_ref_counter_offset);

  auto c = new torch::CudaIPCReceivedData(std::move(basePtr));
  auto sp = std::shared_ptr<void>(
      (void*)c, [ref_counter_handle, ref_counter_offset, device](void* ptr) {
        delete static_cast<torch::CudaIPCReceivedData*>(ptr);
        // Sync default stream to make sure all operations related to the storage is
        // finished (otherwise another process may reuse memory and corrupt
        // data)

        // Ideally all shared memory reference counting could be replaced by
        // sending untriggered CUDA event from the producer to consumer and
        // using this event as the criteria of memory release. However, CUDA (atm 10.1)
        // does not support the creation of untriggered events and performance
        // impact of having thousands of shared events is unknown.

        // TODO: Instead of cudaStreamSynchronize it is possible to add Stream
        // Callback and release counter inside of it (need to check performance impact)
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream(device));

        // We don't want to break existing code, so resource deletion is best
        // effort basis. Exception expected if producer process terminated
        // before consumer released data.
        int flags =
            TH_ALLOCATOR_MAPPED_SHAREDMEM | TH_ALLOCATOR_MAPPED_NOCREATE;
        try {
          auto sptr = THRefcountedMapAllocator::makeDataPtr(
              ref_counter_handle.c_str(),
              flags,
              sizeof(int64_t) * torch::CUDA_IPC_REF_COUNTER_FILE_SIZE,
              nullptr);
          *(static_cast<int64_t*>(sptr.get()) + ref_counter_offset) -= 1;
        } catch (c10::Error& err) {
          // Already warned inside of producer process
        }
      });

  THWStoragePtr base(THWStorage_(newWithDataAndAllocator)(
      LIBRARY_STATE
      THCIpcDeleter::makeDataPtr(std::move(sp), devPtr),
      storage_size, /* allocator */ nullptr));
  base->set_resizable(false);
  base->set_received_cuda(true);

  return THPStorage_(New)(base.release());
  END_HANDLE_TH_ERRORS
}
#endif

// Returns an object that holds a "weak" pointer to the THStorage.  This
// pointer keeps the THStorage struct live, but does not retain the data
// pointer.
//
// NB: This does NOT preserve object identity when you call it multiple times
static PyObject * THPStorage_(weakRef)(PyObject *_self, PyObject *args) {
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  THStorage* storage = self->cdata;
  return PyLong_FromVoidPtr(c10::raw::intrusive_ptr::make_weak(storage));
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(newWithWeakPtr)(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg),
      "_new_with_weak_ptr(): arg must be an 'int'");
  THStorage *weak_storage = (THStorage*)PyLong_AsVoidPtr(arg);
  if (auto* storage = c10::raw::weak_intrusive_ptr::lock(weak_storage)) {
    return THPStorage_(New)(storage);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(freeWeakRef)(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  if (arg == Py_None) {
    Py_RETURN_NONE;
  }
  THPUtils_assert(THPUtils_checkLong(arg),
      "_free_weak_ref(): arg must be an 'int'");
  THStorage *weak_storage = (THStorage*)PyLong_AsVoidPtr(arg);
  c10::raw::weak_intrusive_ptr::decref(weak_storage);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(expired)(PyObject *_unused, PyObject *arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "_expired(): arg must be an 'int'");
  THStorage *weak_storage = (THStorage*)PyLong_AsVoidPtr(arg);
  return PyBool_FromLong(c10::raw::weak_intrusive_ptr::use_count(weak_storage) == 0);
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(sharedFd)(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  THMapAllocator *ctx = nullptr;
#ifndef THC_GENERIC_FILE
  THWStorage *storage = self->cdata;
  ctx = THMapAllocator::fromDataPtr(storage->data_ptr());
#endif

  THPUtils_assert(ctx, "couldn't retrieve a shared file descriptor");
  return THPUtils_packInt32(ctx->fd());
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(isShared)(PyObject *_self, PyObject *noargs)
{
  auto self = (THPStorage*)_self;
#ifdef THC_GENERIC_FILE
  Py_RETURN_TRUE;
#else
  if (THMapAllocator::fromDataPtr(self->cdata->data_ptr()) ||
      THManagedMapAllocator::fromDataPtr(self->cdata->data_ptr())) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
#endif
}

static PyMethodDef THPStorage_(sharingMethods)[] = {
  {"_new_with_weak_ptr", THPStorage_(newWithWeakPtr), METH_O | METH_CLASS, nullptr},
#ifdef THC_GENERIC_FILE
  {"_share_cuda_", THPStorage_(shareCuda), METH_NOARGS, nullptr},
  {"_new_shared_cuda", THPStorage_(newSharedCuda), METH_VARARGS | METH_STATIC, nullptr},
  {"_release_ipc_counter", THPStorage_(releaseIPCCounter), METH_VARARGS | METH_STATIC, nullptr},
#else
  {"_share_fd_", THPStorage_(shareFd), METH_NOARGS, nullptr},
  {"_new_shared_fd", THPStorage_(newSharedFd), METH_VARARGS | METH_STATIC, nullptr},
  {"_new_using_fd", THPStorage_(pyNewFdStorage), METH_VARARGS | METH_STATIC, nullptr},
  {"_share_filename_", THPStorage_(shareFilename), METH_NOARGS, nullptr},
  {"_new_shared_filename", THPStorage_(newSharedFilename), METH_VARARGS | METH_STATIC, nullptr},
  {"_new_using_filename", THPStorage_(pyNewFilenameStorage), METH_VARARGS | METH_STATIC, nullptr},
#endif
  {"_weak_ref", THPStorage_(weakRef), METH_NOARGS, nullptr},
  {"_free_weak_ref", THPStorage_(freeWeakRef), METH_O | METH_STATIC, nullptr},
  {"_expired", THPStorage_(expired), METH_O | METH_STATIC, nullptr},
  {"_shared_decref", THPStorage_(sharedDecref), METH_NOARGS, nullptr},
  {"_shared_incref", THPStorage_(sharedIncref), METH_NOARGS, nullptr},
  {"_get_shared_fd", THPStorage_(sharedFd), METH_NOARGS, nullptr},
  {"is_shared", THPStorage_(isShared), METH_NOARGS, nullptr},
  {nullptr}
};
