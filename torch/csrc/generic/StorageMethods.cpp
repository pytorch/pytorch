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
    return NULL;
  }

  if (count < 0) {
    if ((buffer.len - offset) % sizeof(real) != 0) {
      PyErr_Format(PyExc_ValueError, "buffer size must be a multiple of element size");
      return NULL;
    }
    count = (buffer.len - offset) / sizeof(real);
  }

  if (offset + (count * (Py_ssize_t)sizeof(real)) > buffer.len) {
    PyErr_Format(PyExc_ValueError, "buffer is smaller than requested size");
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
  THP_decodeInt64Buffer(storage->data, src + offset, byte_order, count);
#elif defined(TH_REAL_IS_FLOAT)
  THP_decodeFloatBuffer(storage->data, src + offset, byte_order, count);
#elif defined(TH_REAL_IS_DOUBLE)
  THP_decodeDoubleBuffer(storage->data, src + offset, byte_order, count);
#else
#error "Unknown type"
#endif

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
PyObject * THPStorage_(_share)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THStorage *storage = self->cdata;
  libshm_context *ctx;
  // Storage is already in shared memory, just return a handle
  if (storage->allocator == &THManagedSharedAllocator) {
    ctx = (libshm_context*)storage->allocatorContext;
  } else if (storage->allocator == &THStorageWeakRefAllocator) {
    auto allocator_obj = ((StorageWeakRefAllocator*)storage->allocatorContext);
    ctx = (libshm_context*)allocator_obj->allocatorContext;
  } else {
    Py_BEGIN_ALLOW_THREADS
    std::string handle = "/torch_";
    handle += std::to_string(getpid());
    handle += "_";
    handle += std::to_string(THRandom_random(THPDefaultGenerator->cdata));
    // TODO: retry on collision
    ctx = libshm_context_new(NULL, handle.c_str(),
            TH_ALLOCATOR_MAPPED_SHAREDMEM | TH_ALLOCATOR_MAPPED_EXCLUSIVE);
    THStoragePtr new_storage = THStorage_(newWithAllocator)(storage->size,
            &THManagedSharedAllocator, (void*)ctx);
    THStorage_(copy)(new_storage, storage);
    THStorage_(swap)(storage, new_storage);
    Py_END_ALLOW_THREADS
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
  size.release();

  THPObjectPtr result_tuple = PyTuple_New(2);
  PyTuple_SET_ITEM(result_tuple.get(), 0, tuple.release());
  PyTuple_SET_ITEM(result_tuple.get(), 1, THPStorage_(newWeakObject)(self->cdata));

  return result_tuple.release();
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(_newShared)(THPStorage *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (!args || PyTuple_Size(args) != 3) {
    THPUtils_setError("new_shared expects exactly three arguments");
    return NULL;
  }
  PyObject *_manager_handle = PyTuple_GET_ITEM(args, 0);
  PyObject *_object_handle = PyTuple_GET_ITEM(args, 1);
  PyObject *_size = PyTuple_GET_ITEM(args, 2);
  if (!THPUtils_checkBytes(_manager_handle) || !THPUtils_checkBytes(_object_handle) || !THPUtils_checkLong(_size)) {
    THPUtils_invalidArguments(args, "a handle (string/bytes) and storage size (int)");
    return NULL;
  }
  const char *manager_handle = THPUtils_bytesAsString(_manager_handle);
  const char *object_handle = THPUtils_bytesAsString(_object_handle);
  long size = THPUtils_unpackLong(_size);

  libshm_context *ctx = libshm_context_new(manager_handle, object_handle,
          TH_ALLOCATOR_MAPPED_SHAREDMEM | TH_ALLOCATOR_MAPPED_NOCREATE);
  THStoragePtr storage_guard = THStorage_(newWithAllocator)(size,
          &THManagedSharedAllocator, (void*)ctx);
  THPObjectPtr result = THPStorage_(newObject)(storage_guard);
  THStorage *storage = storage_guard.release();

  THPObjectPtr tuple = PyTuple_New(2);
  PyTuple_SET_ITEM(tuple.get(), 0, result.release());
  PyTuple_SET_ITEM(tuple.get(), 1, THPStorage_(newWeakObject)(storage));
  return tuple.release();
  END_HANDLE_TH_ERRORS
}
#endif

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
  {"_share", (PyCFunction)THPStorage_(_share), METH_NOARGS, NULL},
  {"_new_shared", (PyCFunction)THPStorage_(_newShared), METH_VARARGS | METH_STATIC, NULL},
#endif
  {NULL}
};
