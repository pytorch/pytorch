#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

static PyObject * THPStorage_(size)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  return PyLong_FromLong(THStorage_(size)(LIBRARY_STATE self->cdata));
  END_HANDLE_TH_ERRORS
}

#ifndef THD_GENERIC_FILE
static PyObject * THPStorage_(dataPtr)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(THStorage_(data)(LIBRARY_STATE self->cdata));
  END_HANDLE_TH_ERRORS
}
#endif

static PyObject * THPStorage_(copy_)(PyObject *self, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  return THPCopyMethod(THStorage_(copy_functions), self, args, kwargs);
  END_HANDLE_TH_ERRORS
}

#ifndef THD_GENERIC_FILE
static PyObject * THPStorage_(isPinned)(THPStorage *self)
{
  HANDLE_TH_ERRORS
#if defined(WITH_CUDA)
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, self->cdata->data);
  if (err != cudaSuccess) {
    cudaGetLastError();
    Py_RETURN_FALSE;
  }
  return PyBool_FromLong(attr.memoryType == cudaMemoryTypeHost);
#else
  Py_RETURN_FALSE;
#endif
  END_HANDLE_TH_ERRORS
}
#endif

static PyObject * THPStorage_(elementSize)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  return PyLong_FromLong(THStorage_(elementSize)(LIBRARY_STATE_NOARGS));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(new)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THStoragePtr new_storage(THStorage_(new)(LIBRARY_STATE_NOARGS));
  PyObject *_ret = THPStorage_(New)(new_storage);
  new_storage.release();
  return _ret;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(resize_)(THPStorage *self, PyObject *number_arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(number_arg), "resize_ expects an int, "
      "but got %s", THPUtils_typename(number_arg));
  long newsize = THPUtils_unpackLong(number_arg);
  THStorage_(resize)(LIBRARY_STATE self->cdata, newsize);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(fill_)(THPStorage *self, PyObject *number_arg)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_(checkReal)(number_arg), "fill_ expects %s, "
      "but got %s", THPUtils_typeTraits<real>::python_type_str,
      THPUtils_typename(number_arg));
  THStorage_(fill)(LIBRARY_STATE self->cdata, THPUtils_(unpackReal)(number_arg));
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

#if !defined(THC_GENERIC_FILE) && !defined(THD_GENERIC_FILE)
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
      "offset must be non-negative and no greater than buffer length (%ld), "
      "but got %ld", (long)offset, (long)buffer.len);
    PyBuffer_Release(&buffer);
    return NULL;
  }

  if (count < 0) {
    if ((buffer.len - offset) % sizeof(real) != 0) {
      PyErr_Format(PyExc_ValueError, "buffer size (%ld) must be a multiple "
          "of element size (%ld)", (long)buffer.len, (long)sizeof(real));
      PyBuffer_Release(&buffer);
      return NULL;
    }
    count = (buffer.len - offset) / sizeof(real);
  }

  if (offset + (count * (Py_ssize_t)sizeof(real)) > buffer.len) {
    PyErr_Format(PyExc_ValueError, "buffer has only %ld elements after offset "
        "%ld, but specified a size of %ld", (long)(buffer.len - offset),
        (long)offset, (long)count);
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
#elif defined(TH_REAL_IS_HALF)
  THP_decodeHalfBuffer(storage->data, src + offset, byte_order, count);
#elif defined(TH_REAL_IS_FLOAT)
  THP_decodeFloatBuffer(storage->data, src + offset, byte_order, count);
#elif defined(TH_REAL_IS_DOUBLE)
  THP_decodeDoubleBuffer(storage->data, src + offset, byte_order, count);
#else
#error "Unknown type"
#endif

  PyBuffer_Release(&buffer);
  return (PyObject*)THPStorage_(New)(storage);
  END_HANDLE_TH_ERRORS
}
#endif

#ifndef THD_GENERIC_FILE
PyObject * THPStorage_(writeFile)(THPStorage *self, PyObject *file)
{
  HANDLE_TH_ERRORS
  int fd = PyObject_AsFileDescriptor(file);
  THPUtils_assert(fd != -1, "_write_file couldn't retrieve a file descriptor "
      "from given object");
  THPStorage_(writeFileRaw)(self->cdata, fd);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(newWithFile)(PyObject *_unused, PyObject *file)
{
  HANDLE_TH_ERRORS
  int fd = PyObject_AsFileDescriptor(file);
  THPUtils_assert(fd != -1, "_new_with_file couldn't retrieve a file "
      "descriptor from given object");
  THStorage *storage = THPStorage_(readFileRaw)(fd, nullptr);
  if (storage == nullptr)
    return nullptr;
  PyObject *result = THPStorage_(New)(storage);
  return result;
  END_HANDLE_TH_ERRORS
}

static PyObject *THPStorage_(setFromFile)(THPStorage *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject *file = PyTuple_GET_ITEM(args, 0);
  int fd = PyObject_AsFileDescriptor(file);

  PyObject *offset = PyTuple_GET_ITEM(args, 1);
  if (offset != Py_None) {
    lseek(fd, THPUtils_unpackLong(offset), SEEK_SET);
  }

  THPUtils_assert(fd != -1, "_set_from_file couldn't retrieve a file "
      "descriptor from given object");
  THStorage *storage = THPStorage_(readFileRaw)(fd, self->cdata);
  if (storage == nullptr)
    return nullptr;
  Py_INCREF(self);

  return (PyObject *) self;
  END_HANDLE_TH_ERRORS
}
#endif // !defined(THD_GENERIC_FILE)

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
  THPUtils_assert(THPUtils_checkLong(new_cdata), "given an invalid argument to "
      "_set_cdata - expected an int or long, but got %s",
      THPUtils_typename(new_cdata));
  THStorage *ptr = (THStorage*)PyLong_AsVoidPtr(new_cdata);
  THStorage_(retain)(LIBRARY_STATE ptr);
  THStorage_(free)(LIBRARY_STATE self->cdata);
  self->cdata = ptr;
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

#ifndef THD_GENERIC_FILE
PyObject * THPStorage_(_rootStorage)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  if (!(self->cdata->flag & TH_STORAGE_VIEW)) {
    return Py_BuildValue("(ON)", self, PyLong_FromLong(0));
  }
  THStorage *root = self->cdata;
  while (root->flag & TH_STORAGE_VIEW)
    root = root->view;
  size_t offset = self->cdata->data - root->data;
  THStorage_(retain)(LIBRARY_STATE root);
  THPObjectPtr storage(THPStorage_(New)(root));
  PyObject *result = Py_BuildValue("(NN)", storage.get(), PyLong_FromLong(offset));
  storage.release();
  return result;
  END_HANDLE_TH_ERRORS
}
#endif

static PyMethodDef THPStorage_(methods)[] = {
  {"copy_", (PyCFunction)THPStorage_(copy_), METH_VARARGS | METH_KEYWORDS, NULL},
  {"element_size", (PyCFunction)THPStorage_(elementSize), METH_NOARGS, NULL},
  {"fill_", (PyCFunction)THPStorage_(fill_), METH_O, NULL},
  {"new", (PyCFunction)THPStorage_(new), METH_NOARGS, NULL},
  {"resize_", (PyCFunction)THPStorage_(resize_), METH_O, NULL},
  {"size", (PyCFunction)THPStorage_(size), METH_NOARGS, NULL},
#ifndef THD_GENERIC_FILE
  {"data_ptr", (PyCFunction)THPStorage_(dataPtr), METH_NOARGS, NULL},
  {"is_pinned", (PyCFunction)THPStorage_(isPinned), METH_NOARGS, NULL},
  {"_write_file", (PyCFunction)THPStorage_(writeFile), METH_O, NULL},
  {"_new_with_file", (PyCFunction)THPStorage_(newWithFile), METH_O | METH_STATIC, NULL},
  {"_set_from_file", (PyCFunction)THPStorage_(setFromFile), METH_VARARGS, NULL},
#endif // !defined(THD_GENERIC_FILE)
#if !defined(THC_GENERIC_FILE) && !defined(THD_GENERIC_FILE)
  {"from_buffer", (PyCFunction)THPStorage_(fromBuffer), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
#endif
#ifdef THC_GENERIC_FILE
  {"get_device", (PyCFunction)THPStorage_(getDevice), METH_NOARGS, NULL},
#endif
  {"_set_cdata", (PyCFunction)THPStorage_(_setCdata), METH_O, NULL},
#ifndef THD_GENERIC_FILE
  {"_root_storage", (PyCFunction)THPStorage_(_rootStorage), METH_NOARGS, NULL},
#endif
  {NULL}
};
