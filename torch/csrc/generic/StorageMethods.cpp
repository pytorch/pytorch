static PyObject * THPStorage_(size)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  return PyLong_FromLong(THStorage_(size)(LIBRARY_STATE self->cdata));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(elementSize)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  return PyLong_FromLong(THStorage_(elementSize)(LIBRARY_STATE_NOARGS));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(retain)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  if (self->cdata)
    THStorage_(retain)(LIBRARY_STATE self->cdata);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(free)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THStorage_(free)(LIBRARY_STATE self->cdata);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(new)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THStoragePtr new_storage = THStorage_(new)(LIBRARY_STATE_NOARGS);
  PyObject *_ret = THPStorage_(newObject)(new_storage);
  new_storage.release();
  return _ret;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(resize_)(THPStorage *self, PyObject *number_arg)
{
  HANDLE_TH_ERRORS
  long newsize;
  if (!THPUtils_getLong(number_arg, &newsize))
    return NULL;
  THStorage_(resize)(LIBRARY_STATE self->cdata, newsize);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(fill_)(THPStorage *self, PyObject *number_arg)
{
  HANDLE_TH_ERRORS
  real rvalue;
  if (!THPUtils_(parseReal)(number_arg, &rvalue))
    return NULL;
  THStorage_(fill)(LIBRARY_STATE self->cdata, rvalue);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

#ifndef THC_GENERIC_FILE
static PyObject * THPStorage_(fromBuffer)(PyObject *_unused, PyObject *args, PyObject *keywds)
{
  HANDLE_TH_ERRORS
  PyObject *obj = NULL;
  const char* byte_order_str = NULL;
  Py_ssize_t count = -1, offset = 0;
  Py_buffer buffer;
  static char *kwlist[] = {"buffer", "byte_order", "count", "offset", NULL};
  const char* argtypes;
#if defined(TH_REAL_IS_BYTE) || defined(TH_REAL_IS_CHAR)
  argtypes = "O|snn";
#else
  argtypes = "Os|nn";
#endif

  if (!PyArg_ParseTupleAndKeywords(args, keywds, argtypes, kwlist,
        &obj, &byte_order_str, &count, &offset)) {
    return NULL;
  }

#if !(defined(TH_REAL_IS_BYTE) || defined(TH_REAL_IS_CHAR))
  THPByteOrder byte_order;
  if (strcmp(byte_order_str, "native") == 0) {
    byte_order = THP_nativeByteOrder();
  } else if (strcmp(byte_order_str, "big") == 0) {
    byte_order = THP_BIG_ENDIAN;
  } else if (strcmp(byte_order_str, "little") == 0) {
    byte_order = THP_LITTLE_ENDIAN;
  } else {
    PyErr_Format(PyExc_ValueError,
      "invalid byte_order '%s' (expected 'big', 'little', or 'native')",
      byte_order_str);
    return NULL;
  }
#endif

  if (PyObject_GetBuffer(obj, &buffer, PyBUF_SIMPLE) < 0)
    return NULL;

  if (offset < 0 || offset > buffer.len) {
    PyErr_Format(PyExc_ValueError,
      "offset must be non-negative and no greater than buffer length (%ld)",
      (long) buffer.len);
    PyBuffer_Release(&buffer);
    return NULL;
  }

  if (count < 0) {
    if ((buffer.len - offset) % sizeof(real) != 0) {
      PyErr_Format(PyExc_ValueError, "buffer size must be a multiple of element size");
      PyBuffer_Release(&buffer);
      return NULL;
    }
    count = (buffer.len - offset) / sizeof(real);
  }

  if (offset + (count * (Py_ssize_t)sizeof(real)) > buffer.len) {
    PyErr_Format(PyExc_ValueError, "buffer is smaller than requested size");
    PyBuffer_Release(&buffer);
    return NULL;
  }

  uint8_t* src = (uint8_t*) buffer.buf;
  THStorage* storage = THStorage_(newWithSize)(count);

#if defined(TH_REAL_IS_BYTE) || defined(TH_REAL_IS_CHAR)
  memcpy(storage->data, src + offset, count);
#elif defined(TH_REAL_IS_SHORT)
  THP_decodeInt16Buffer(storage->data, src + offset, byte_order, count);
#elif defined(TH_REAL_IS_INT)
  THP_decodeInt32Buffer(storage->data, src + offset, byte_order, count);
#elif defined(TH_REAL_IS_LONG)
  // TODO: remove the cast
  THP_decodeInt64Buffer((int64_t*) storage->data, src + offset, byte_order, count);
#elif defined(TH_REAL_IS_FLOAT)
  THP_decodeFloatBuffer(storage->data, src + offset, byte_order, count);
#elif defined(TH_REAL_IS_DOUBLE)
  THP_decodeDoubleBuffer(storage->data, src + offset, byte_order, count);
#else
#error "Unknown type"
#endif

  PyBuffer_Release(&buffer);
  return (PyObject*)THPStorage_(newObject)(storage);
  END_HANDLE_TH_ERRORS
}
#endif

PyObject * THPStorage_(writeFile)(THPStorage *self, PyObject *file)
{
  HANDLE_TH_ERRORS
  int fd = PyObject_AsFileDescriptor(file);
  if (fd == -1) {
    THPUtils_setError("_write_file couln't retrieve file descriptor from given object");
    return NULL;
  }
  THPStorage_(writeFileRaw)(self->cdata, fd);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(newWithFile)(PyObject *_unused, PyObject *file)
{
  HANDLE_TH_ERRORS
  int fd = PyObject_AsFileDescriptor(file);
  if (fd == -1) {
    THPUtils_setError("_new_with_file couln't retrieve file descriptor from given object");
    return NULL;
  }
  THStoragePtr storage = THPStorage_(readFileRaw)(fd);
  PyObject *result = THPStorage_(newObject)(storage);
  storage.release();
  return result;
  END_HANDLE_TH_ERRORS
}

#ifndef THC_GENERIC_FILE
// TODO: move this somewhere - we only need one version
static std::string THPStorage_(__newHandle)() {
  std::string handle = "/torch_";
  handle += std::to_string(getpid());
  handle += "_";
  handle += std::to_string(THRandom_random(THPDefaultGenerator->cdata));
  return handle;
}

PyObject * THPStorage_(_sharedDecref)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  libshm_context *ctx = NULL;
  THStorage *storage = self->cdata;
  if (storage->allocator == &THManagedSharedAllocator) {
    ctx = (libshm_context*)storage->allocatorContext;
  } else if (storage->allocator == &THStorageWeakRefAllocator) {
    auto allocator_obj = ((StorageWeakRefAllocator*)storage->allocatorContext);
    if (allocator_obj->allocator == &THManagedSharedAllocator)
      ctx = (libshm_context*)allocator_obj->allocatorContext;
  }
  if (ctx)
    THRefcountedMapAllocator_decref(ctx->th_context, storage->data);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(_sharedIncref)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  libshm_context *ctx = NULL;
  THStorage *storage = self->cdata;
  if (storage->allocator == &THManagedSharedAllocator) {
    ctx = (libshm_context*)storage->allocatorContext;
  } else if (storage->allocator == &THStorageWeakRefAllocator) {
    auto allocator_obj = ((StorageWeakRefAllocator*)storage->allocatorContext);
    if (allocator_obj->allocator == &THManagedSharedAllocator)
      ctx = (libshm_context*)allocator_obj->allocatorContext;
  }
  if (ctx)
    THRefcountedMapAllocator_incref(ctx->th_context, storage->data);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(_share_filename)(THPStorage *self)
{
  THStorage *storage = self->cdata;
  libshm_context *ctx;
  // Storage is already in shared memory, just return a handle
  if (storage->allocator == &THManagedSharedAllocator) {
    ctx = (libshm_context*)storage->allocatorContext;
  } else if (storage->allocator == &THStorageWeakRefAllocator) {
    auto allocator_obj = ((StorageWeakRefAllocator*)storage->allocatorContext);
    ctx = (libshm_context*)allocator_obj->allocatorContext;
  } else {
    // TODO: retry on collision
    // TODO: free GIL - but remember to reacquire it when an exception is thrown
    std::string handle = THPStorage_(__newHandle)();
    ctx = libshm_context_new(NULL, handle.c_str(),
            TH_ALLOCATOR_MAPPED_SHAREDMEM | TH_ALLOCATOR_MAPPED_EXCLUSIVE);
    THStoragePtr new_storage = THStorage_(newWithAllocator)(storage->size,
            &THManagedSharedAllocator, (void*)ctx);
    THStorage_(copy)(new_storage, storage);
    THStorage_(swap)(storage, new_storage);
  }

  THPObjectPtr manager_handle =
    THPUtils_bytesFromString(ctx->manager_handle);
  THPObjectPtr storage_handle =
    THPUtils_bytesFromString(THMapAllocatorContext_filename(ctx->th_context));
  THPObjectPtr size = PyLong_FromLong(storage->size);

  THPObjectPtr tuple = PyTuple_New(3);
  PyTuple_SET_ITEM(tuple.get(), 0, manager_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 1, storage_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 2, size.release());
  return tuple.release();
}

PyObject * THPStorage_(_share_fd)(THPStorage *self)
{
  THStorage *storage = self->cdata;
  THMapAllocatorContext *ctx;
  // Storage is already in shared memory, just return a handle
  if (storage->allocator == &THMapAllocator) {
    ctx = (THMapAllocatorContext*)storage->allocatorContext;
  } else if (storage->allocator == &THStorageWeakRefAllocator) {
    auto allocator_obj = ((StorageWeakRefAllocator*)storage->allocatorContext);
    ctx = (THMapAllocatorContext*)allocator_obj->allocatorContext;
  } else {
    int flags = TH_ALLOCATOR_MAPPED_SHAREDMEM |
                TH_ALLOCATOR_MAPPED_EXCLUSIVE |
                TH_ALLOCATOR_MAPPED_KEEPFD |
                TH_ALLOCATOR_MAPPED_UNLINK;
    std::string handle = THPStorage_(__newHandle)();
    ctx = THMapAllocatorContext_new(handle.c_str(), flags);
    THStoragePtr new_storage = THStorage_(newWithAllocator)(storage->size,
            &THMapAllocator, (void*)ctx);
    THStorage_(copy)(new_storage, storage);
    THStorage_(swap)(storage, new_storage);
  }

  THPObjectPtr storage_handle = PyLong_FromLong(THMapAllocatorContext_fd(ctx));
  THPObjectPtr size = PyLong_FromLong(storage->size);

  THPObjectPtr tuple = PyTuple_New(2);
  PyTuple_SET_ITEM(tuple.get(), 0, storage_handle.release());
  PyTuple_SET_ITEM(tuple.get(), 1, size.release());
  return tuple.release();
}

PyObject * THPStorage_(_share)(THPStorage *self, PyObject *use_fd)
{
  HANDLE_TH_ERRORS
  THPObjectPtr result_tuple = PyTuple_New(2);

  if (use_fd == Py_False) {
    PyTuple_SET_ITEM(result_tuple.get(), 0, THPStorage_(_share_filename)(self));
    PyTuple_SET_ITEM(result_tuple.get(), 1, THPStorage_(newWeakObject)(self->cdata));
  } else {
    PyTuple_SET_ITEM(result_tuple.get(), 0, THPStorage_(_share_fd)(self));
    PyTuple_SET_ITEM(result_tuple.get(), 1, THPStorage_(newWeakObject)(self->cdata));
  }

  return result_tuple.release();
  END_HANDLE_TH_ERRORS
}

THStorage * THPStorage_(_newShared_filename)(PyObject *args)
{
  PyObject *_manager_handle = PyTuple_GET_ITEM(args, 0);
  PyObject *_object_handle = PyTuple_GET_ITEM(args, 1);
  PyObject *_size = PyTuple_GET_ITEM(args, 2);
  if (!THPUtils_checkBytes(_manager_handle) || !THPUtils_checkBytes(_object_handle) || !THPUtils_checkLong(_size)) {
    THPUtils_invalidArguments(args, "_new_shared in file system mode", 1, "a handle (string/bytes) and storage size (int)");
    return NULL;
  }
  const char *manager_handle = THPUtils_bytesAsString(_manager_handle);
  const char *object_handle = THPUtils_bytesAsString(_object_handle);
  long size = THPUtils_unpackLong(_size);

  libshm_context *ctx = libshm_context_new(manager_handle, object_handle,
          TH_ALLOCATOR_MAPPED_SHAREDMEM | TH_ALLOCATOR_MAPPED_NOCREATE);
  THStorage *storage = THStorage_(newWithAllocator)(size,
          &THManagedSharedAllocator, (void*)ctx);
  return storage;
}

THStorage * THPStorage_(_newShared_fd)(PyObject *args)
{
  PyObject *_tmp_fd = PyTuple_GET_ITEM(args, 0);
  PyObject *_size = PyTuple_GET_ITEM(args, 1);
  if (!THPUtils_checkLong(_tmp_fd) || !THPUtils_checkLong(_size)) {
    THPUtils_invalidArguments(args, "_new_shared in file descriptor mode", 1, "a handle (string/bytes) and storage size (int)");
    return NULL;
  }
  int fd;
  long tmp_fd = THPUtils_unpackLong(_tmp_fd);
  long size = THPUtils_unpackLong(_size);
  if ((fd = dup(tmp_fd)) == -1) {
    THPUtils_setError("could not duplicate a shared memory file descriptor");
    return NULL;
  }

  int flags = TH_ALLOCATOR_MAPPED_SHAREDMEM |
              TH_ALLOCATOR_MAPPED_NOCREATE |
              TH_ALLOCATOR_MAPPED_KEEPFD |
              TH_ALLOCATOR_MAPPED_FROMFD;
  THMapAllocatorContext *ctx = THMapAllocatorContext_newWithFd(NULL, fd, flags);
  THStorage *storage = THStorage_(newWithAllocator)(size, &THMapAllocator,
      (void*)ctx);
  return storage;
}

PyObject * THPStorage_(_newShared)(THPStorage *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (!args || PyTuple_Size(args) < 2 || PyTuple_Size(args) > 3) {
    THPUtils_setError("new_shared expects two or three arguments");
    return NULL;
  }
  THStoragePtr storage_guard;
  if (PyTuple_Size(args) == 3) {
    storage_guard = THPStorage_(_newShared_filename)(args);
  } else {
    storage_guard = THPStorage_(_newShared_fd)(args);
  }
  THPObjectPtr result = THPStorage_(newObject)(storage_guard.get());
  THStorage *storage = storage_guard.release();

  THPObjectPtr tuple = PyTuple_New(2);
  PyTuple_SET_ITEM(tuple.get(), 0, result.release());
  PyTuple_SET_ITEM(tuple.get(), 1, THPStorage_(newWeakObject)(storage));
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(_sharedFd)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THMapAllocatorContext *ctx = NULL;
  THStorage *storage = self->cdata;
  if (storage->allocator == &THMapAllocator) {
    ctx = (THMapAllocatorContext*)storage->allocatorContext;
  } else if (storage->allocator == &THStorageWeakRefAllocator) {
    auto allocator_obj = ((StorageWeakRefAllocator*)storage->allocatorContext);
    if (allocator_obj->allocator == &THMapAllocator) {
      ctx = (THMapAllocatorContext*)allocator_obj->allocatorContext;
    }
  }

  if (!ctx) {
    THPUtils_setError("can't retrieve shared file descriptor");
    return NULL;
  }
  return PyLong_FromLong(THMapAllocatorContext_fd(ctx));
  END_HANDLE_TH_ERRORS
}
#endif

#ifdef THC_GENERIC_FILE
PyObject * THPStorage_(getDevice)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  return PyLong_FromLong(THCStorage_(getDevice)(LIBRARY_STATE self->cdata));
  END_HANDLE_TH_ERRORS
}
#endif

PyObject * THPStorage_(_setCdata)(THPStorage *self, PyObject *new_cdata)
{
  HANDLE_TH_ERRORS
  if (!THPUtils_checkLong(new_cdata)) {
    THPUtils_setError("invalid argument to _set_cdata - expected an int or long");
    return NULL;
  }
  THStorage *ptr = (THStorage*)PyLong_AsVoidPtr(new_cdata);
  THStorage_(retain)(LIBRARY_STATE ptr);
  THStorage_(free)(LIBRARY_STATE self->cdata);
  self->cdata = ptr;
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyMethodDef THPStorage_(methods)[] = {
  {"elementSize", (PyCFunction)THPStorage_(elementSize), METH_NOARGS, NULL},
  {"fill_", (PyCFunction)THPStorage_(fill_), METH_O, NULL},
  {"free", (PyCFunction)THPStorage_(free), METH_NOARGS, NULL},
  {"new", (PyCFunction)THPStorage_(new), METH_NOARGS, NULL},
  {"resize_", (PyCFunction)THPStorage_(resize_), METH_O, NULL},
  {"retain", (PyCFunction)THPStorage_(retain), METH_NOARGS, NULL},
  {"size", (PyCFunction)THPStorage_(size), METH_NOARGS, NULL},
  {"_write_file", (PyCFunction)THPStorage_(writeFile), METH_O, NULL},
  {"_new_with_file", (PyCFunction)THPStorage_(newWithFile), METH_O | METH_STATIC, NULL},
#ifndef THC_GENERIC_FILE
  {"from_buffer", (PyCFunction)THPStorage_(fromBuffer), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"_share", (PyCFunction)THPStorage_(_share), METH_O, NULL},
  {"_new_shared", (PyCFunction)THPStorage_(_newShared), METH_VARARGS | METH_STATIC, NULL},
  {"_get_shared_fd", (PyCFunction)THPStorage_(_sharedFd), METH_NOARGS, NULL},
  {"_shared_decref", (PyCFunction)THPStorage_(_sharedDecref), METH_NOARGS, NULL},
  {"_shared_incref", (PyCFunction)THPStorage_(_sharedIncref), METH_NOARGS, NULL},
#endif
#ifdef THC_GENERIC_FILE
  {"getDevice", (PyCFunction)THPStorage_(getDevice), METH_NOARGS, NULL},
#endif
  {"_set_cdata", (PyCFunction)THPStorage_(_setCdata), METH_O, NULL},
  {NULL}
};
