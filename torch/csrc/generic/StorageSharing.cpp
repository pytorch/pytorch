#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAGuard.h>
#endif

#include <random>

static PyObject * THPStorage_(sharedDecref)(THPStorage *self)
{
  HANDLE_TH_ERRORS
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

static PyObject * THPStorage_(sharedIncref)(THPStorage *self)
{
  HANDLE_TH_ERRORS
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

static PyObject * THPStorage_(shareFilename)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THWStorage *storage = self->cdata;
  THManagedMapAllocator *ctx;
  // Storage is already in shared memory, just return a handle
  if ((ctx = THManagedMapAllocator::fromDataPtr(storage->data_ptr()))) {
    // done
  } else {
    // TODO: retry on collision
    // TODO: free GIL - but remember to reacquire it when an exception is thrown
    THWStoragePtr new_storage(THPStorage_(newFilenameStorage)(storage->numel()));
    THWStorage_(copy)(new_storage, storage);
    THWStorage_(swap)(storage, new_storage);
    ctx = THManagedMapAllocator::fromDataPtr(storage->data_ptr());
    AT_ASSERT(ctx);
  }

  THPObjectPtr manager_handle(PyBytes_FromString(ctx->manager_handle()));
  if (!manager_handle) return nullptr;
  THPObjectPtr storage_handle(PyBytes_FromString(ctx->filename()));
  if (!storage_handle) return nullptr;
  THPObjectPtr size(PyLong_FromLong(storage->numel()));
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

static PyObject * THPStorage_(shareFd)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THWStorage *storage = self->cdata;
  THMapAllocator *ctx;
  // Storage is already in shared memory, just return a handle
  if ((ctx = THMapAllocator::fromDataPtr(storage->data_ptr()))) {
    // done
  } else {
    THWStoragePtr new_storage(THPStorage_(newFdStorage)(storage->numel()));
    THWStorage_(copy)(new_storage, storage);
    THWStorage_(swap)(storage, new_storage);
    ctx = THMapAllocator::fromDataPtr(storage->data_ptr());
    AT_ASSERT(ctx);
  }

  THPObjectPtr storage_handle(PyLong_FromLong(ctx->fd()));
  if (!storage_handle) return nullptr;
  THPObjectPtr size(PyLong_FromLong(storage->numel()));
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

static PyObject * THPStorage_(shareCuda)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THWStorage *storage = self->cdata;
  at::DeviceGuard device_guard(storage->device());
  THPObjectPtr tuple(PyTuple_New(4));
  THPObjectPtr device(PyLong_FromLong(storage->device().index()));
  THPObjectPtr _handle(Py_None);
  Py_INCREF(Py_None);
  THPObjectPtr size_bytes(PyLong_FromLong(storage->numel() * sizeof(scalar_t)));
  THPObjectPtr _offset_bytes(PyLong_FromLong(0));
  if (THWStorage_(data)(LIBRARY_STATE storage)) {
    size_t base_size;
    void *base_ptr = THCCachingAllocator_getBaseAllocation(THWStorage_(data)(LIBRARY_STATE storage), &base_size);
    ptrdiff_t offset_bytes = (char*)storage->data<scalar_t>() - (char*)base_ptr;

    cudaIpcMemHandle_t handle;
    THCudaCheck(cudaIpcGetMemHandle(&handle, base_ptr));

    _handle = PyBytes_FromStringAndSize((char *)&handle, CUDA_IPC_HANDLE_SIZE);
    _offset_bytes = PyLong_FromSsize_t((Py_ssize_t)offset_bytes);
  }

  if (!tuple || !device || !_handle || !size_bytes || !_offset_bytes) {
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
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(newSharedCuda)(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 4, "tuple of 4 items expected");
  PyObject *_device = PyTuple_GET_ITEM(args, 0);
  PyObject *_handle = PyTuple_GET_ITEM(args, 1);
  PyObject *_size_bytes = PyTuple_GET_ITEM(args, 2);
  PyObject *_offset_bytes = PyTuple_GET_ITEM(args, 3);
  if (!(THPUtils_checkLong(_device) && THPUtils_checkLong(_size_bytes)
      && (_handle != Py_None && PyBytes_Check(_handle))
      && THPUtils_checkLong(_offset_bytes))) {
    THPUtils_invalidArguments(args, nullptr, "_new_shared in CUDA mode", 1,
        "(int device, bytes handle, int storage_size_bytes, int storage_offset_bytes)");
    return nullptr;
  }

  // Storage constructor requires size in numel.
  size_t storage_size = (size_t)THPUtils_unpackLong(_size_bytes) / sizeof(scalar_t);
  ptrdiff_t storage_offset_bytes = (ptrdiff_t)THPUtils_unpackLong(_offset_bytes);

  int64_t device = THPUtils_unpackLong(_device);
  at::cuda::CUDAGuard device_guard(device);

  char *buffer;
  Py_ssize_t handle_size;
  if (PyBytes_AsStringAndSize(_handle, &buffer, &handle_size) == -1) {
    return nullptr;
  }
  THPUtils_assert(handle_size == CUDA_IPC_HANDLE_SIZE, "incorrect handle size");
  std::string s_handle = std::string(buffer, handle_size);
  std::shared_ptr<void> basePtr = THCCaching_CUDAIpcDevptr(s_handle);

  // Offset the basePtr to reconstruct the real storage
  // devPtr = basePtr + storage_offset
  void* devPtr = basePtr.get();
  devPtr = (char*)devPtr + storage_offset_bytes;

  THWStoragePtr base(THWStorage_(newWithDataAndAllocator)(
      LIBRARY_STATE
      THCIpcDeleter::makeDataPtr(std::move(basePtr), devPtr),
      storage_size, /* allocator */ nullptr));
  base->set_resizable(false);

  return THPStorage_(New)(base.release());
  END_HANDLE_TH_ERRORS
}
#endif

// Returns an object that holds a "weak" pointer to the THStorage.  This
// pointer keeps the THStorage struct live, but does not retain the data
// pointer.
//
// NB: This does NOT preserve object identity when you call it multiple times
static PyObject * THPStorage_(weakRef)(THPStorage *self, PyObject *args) {
  HANDLE_TH_ERRORS
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

PyObject * THPStorage_(sharedFd)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THMapAllocator *ctx = nullptr;
#ifndef THC_GENERIC_FILE
  THWStorage *storage = self->cdata;
  ctx = THMapAllocator::fromDataPtr(storage->data_ptr());
#endif

  THPUtils_assert(ctx, "couldn't retrieve a shared file descriptor");
  return PyLong_FromLong(ctx->fd());
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(isShared)(THPStorage *self)
{
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
  {"_new_with_weak_ptr", (PyCFunction)THPStorage_(newWithWeakPtr), METH_O | METH_CLASS, nullptr},
#ifdef THC_GENERIC_FILE
  {"_share_cuda_", (PyCFunction)THPStorage_(shareCuda), METH_NOARGS, nullptr},
  {"_new_shared_cuda", (PyCFunction)THPStorage_(newSharedCuda), METH_VARARGS | METH_STATIC, nullptr},
#else
  {"_share_fd_", (PyCFunction)THPStorage_(shareFd), METH_NOARGS, nullptr},
  {"_new_shared_fd", (PyCFunction)THPStorage_(newSharedFd), METH_VARARGS | METH_STATIC, nullptr},
  {"_new_using_fd", (PyCFunction)THPStorage_(pyNewFdStorage), METH_VARARGS | METH_STATIC, nullptr},
  {"_share_filename_", (PyCFunction)THPStorage_(shareFilename), METH_NOARGS, nullptr},
  {"_new_shared_filename", (PyCFunction)THPStorage_(newSharedFilename), METH_VARARGS | METH_STATIC, nullptr},
  {"_new_using_filename", (PyCFunction)THPStorage_(pyNewFilenameStorage), METH_VARARGS | METH_STATIC, nullptr},
#endif
  {"_weak_ref", (PyCFunction)THPStorage_(weakRef), METH_NOARGS, nullptr},
  {"_free_weak_ref", (PyCFunction)THPStorage_(freeWeakRef), METH_O | METH_STATIC, nullptr},
  {"_expired", (PyCFunction)THPStorage_(expired), METH_O | METH_STATIC, nullptr},
  {"_shared_decref", (PyCFunction)THPStorage_(sharedDecref), METH_NOARGS, nullptr},
  {"_shared_incref", (PyCFunction)THPStorage_(sharedIncref), METH_NOARGS, nullptr},
  {"_get_shared_fd", (PyCFunction)THPStorage_(sharedFd), METH_NOARGS, nullptr},
  {"is_shared", (PyCFunction)THPStorage_(isShared), METH_NOARGS, nullptr},
  {nullptr}
};
