#include <ATen/ATen.h>
#include <ATen/MapAllocator.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_numbers.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef _MSC_VER
#define LSEEK _lseeki64
#else
#define LSEEK lseek
#endif

static PyObject * THPStorage_(size)(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  return THPUtils_packUInt64(self->cdata->nbytes() / sizeof(scalar_t));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(dataPtr)(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  return PyLong_FromVoidPtr(THWStorage_(data)(LIBRARY_STATE self->cdata));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(copy_)(PyObject *self, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  return THPStorageCopyMethod(THWStorage_(copy_functions), self, args, kwargs);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(isPinned)(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
#if defined(USE_CUDA)
  return PyBool_FromLong(at::globalContext().isPinnedPtr(THWStorage_(data)(LIBRARY_STATE self->cdata)));
#else
  Py_RETURN_FALSE;
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(elementSize)(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  return THPUtils_packInt64(THWStorage_(elementSize)(LIBRARY_STATE_NOARGS));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(new)(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  THWStoragePtr new_storage(THWStorage_(new)(LIBRARY_STATE_NOARGS));
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  PyObject *_ret = THPStorage_(New)(new_storage);
  new_storage.release();
  return _ret;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(resize_)(PyObject *_self, PyObject *number_arg)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  THPUtils_assert(THPUtils_checkLong(number_arg), "resize_ expects an int, "
      "but got %s", THPUtils_typename(number_arg));
  int64_t newsize = THPUtils_unpackLong(number_arg);
  THWStorage_(resizeBytes)(
      LIBRARY_STATE self->cdata, newsize * sizeof(scalar_t));
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(fill_)(PyObject *_self, PyObject *number_arg)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  THPUtils_assert(THPUtils_(checkReal)(number_arg), "fill_ expects %s, "
      "but got %s", THPUtils_typeTraits<scalar_t>::python_type_str,
      THPUtils_typename(number_arg));
  THWStorage_(fill)(LIBRARY_STATE self->cdata, THPUtils_(unpackReal)(number_arg));
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

#if !defined(THC_GENERIC_FILE)
static PyObject * THPStorage_(fromBuffer)(PyObject *_unused, PyObject *args, PyObject *keywds)
{
  HANDLE_TH_ERRORS
  PyObject *obj = nullptr;
  const char* byte_order_str = nullptr;
  Py_ssize_t count = -1, offset = 0;
  Py_buffer buffer = {};
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,clang-diagnostic-writable-strings)
  static char *kwlist[] = {"buffer", "byte_order", "count", "offset", nullptr};
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const char* argtypes;
#if defined(TH_REAL_IS_BYTE) || defined(TH_REAL_IS_CHAR)
  argtypes = "O|snn";
#else
  argtypes = "Os|nn";
#endif

  if (!PyArg_ParseTupleAndKeywords(args, keywds, argtypes, kwlist,
        &obj, &byte_order_str, &count, &offset)) {
    return nullptr;
  }

#if !(defined(TH_REAL_IS_BYTE) || defined(TH_REAL_IS_CHAR))
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  torch::utils::THPByteOrder byte_order;
  if (strcmp(byte_order_str, "native") == 0) {
    byte_order = torch::utils::THP_nativeByteOrder();
  } else if (strcmp(byte_order_str, "big") == 0) {
    byte_order = torch::utils::THP_BIG_ENDIAN;
  } else if (strcmp(byte_order_str, "little") == 0) {
    byte_order = torch::utils::THP_LITTLE_ENDIAN;
  } else {
    PyErr_Format(PyExc_ValueError,
      "invalid byte_order '%s' (expected 'big', 'little', or 'native')",
      byte_order_str);
    return nullptr;
  }
#endif

  if (PyObject_GetBuffer(obj, &buffer, PyBUF_SIMPLE) < 0)
    return nullptr;

  if (offset < 0 || offset > buffer.len) {
    PyErr_SetString(PyExc_ValueError, fmt::format(
      "offset must be non-negative and no greater than buffer length ({}) , but got {}",
      offset, buffer.len));
    PyBuffer_Release(&buffer);
    return nullptr;
  }

  if (count < 0) {
    if ((buffer.len - offset) % sizeof(scalar_t) != 0) {
      PyErr_SetString(PyExc_ValueError, fmt::format(
         "buffer size ({}) must be a multiple of element size ({})",
         buffer.len, sizeof(scalar_t)));
      PyBuffer_Release(&buffer);
      return nullptr;
    }
    count = (buffer.len - offset) / sizeof(scalar_t);
  }

  if (offset + (count * (Py_ssize_t)sizeof(scalar_t)) > buffer.len) {
    PyErr_SetString(PyExc_ValueError, fmt::format(
        "buffer has only {} elements after offset {}, but specified a size of {}",
        buffer.len - offset, offset, count));
    PyBuffer_Release(&buffer);
    return nullptr;
  }

  uint8_t* src = (uint8_t*) buffer.buf;
  THWStorage* storage = THWStorage_(newWithSize)(count);

#if defined(TH_REAL_IS_BYTE) || defined(TH_REAL_IS_CHAR)
  memcpy(THWStorage_(data)(storage), src + offset, count);
#elif defined(TH_REAL_IS_BOOL)
  // Because of ASAN checks, that are failing in the THStorage.cpp whenever
  // we are trying to get a value which is not 0 or 1, we have to manually
  // convert original values to boolean ones.
  torch::utils::THP_decodeBoolBuffer(
      THWStorage_(data)(storage), src + offset, byte_order, count);
#elif defined(TH_REAL_IS_SHORT)
  torch::utils::THP_decodeInt16Buffer(
      THWStorage_(data)(storage), src + offset, byte_order, count);
#elif defined(TH_REAL_IS_INT)
  torch::utils::THP_decodeInt32Buffer(
      THWStorage_(data)(storage), src + offset, byte_order, count);
#elif defined(TH_REAL_IS_LONG)
  // TODO: remove the cast
  torch::utils::THP_decodeInt64Buffer(
      (int64_t*)THWStorage_(data)(storage), src + offset, byte_order, count);
#elif defined(TH_REAL_IS_HALF)
  torch::utils::THP_decodeHalfBuffer(
      THWStorage_(data)(storage), src + offset, byte_order, count);
#elif defined(TH_REAL_IS_BFLOAT16)
  torch::utils::THP_decodeBFloat16Buffer(
      THWStorage_(data)(storage), src + offset, byte_order, count);
#elif defined(TH_REAL_IS_FLOAT)
  torch::utils::THP_decodeFloatBuffer(
      THWStorage_(data)(storage), src + offset, byte_order, count);
#elif defined(TH_REAL_IS_DOUBLE)
  torch::utils::THP_decodeDoubleBuffer(
      THWStorage_(data)(storage), src + offset, byte_order, count);
#elif defined(TH_REAL_IS_COMPLEXFLOAT)
  torch::utils::THP_decodeComplexFloatBuffer(
      THWStorage_(data)(storage), src + offset, byte_order, count);
#elif defined(TH_REAL_IS_COMPLEXDOUBLE)
  torch::utils::THP_decodeComplexDoubleBuffer(
      THWStorage_(data)(storage), src + offset, byte_order, count);
#else
#error "Unknown type"
#endif

  PyBuffer_Release(&buffer);
  return (PyObject*)THPStorage_(New)(storage);
  END_HANDLE_TH_ERRORS
}
#endif

static PyObject * THPStorage_(fromFile)(PyObject *_unused, PyObject *args, PyObject *keywds)
{
  HANDLE_TH_ERRORS
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const char *filename;
  Py_ssize_t size = 0;
  int shared = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,clang-diagnostic-writable-strings)
  static char *kwlist[] = {"filename", "shared", "size", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|in", kwlist,
              &filename, &shared, &size)) {
    return nullptr;
  }
  if (shared)
    shared = at::ALLOCATOR_MAPPED_SHARED;
  THWStorage *storage = THWStorage_(newWithMapping)(LIBRARY_STATE filename, size, shared);
  return (PyObject*)THPStorage_(New)(storage);
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(writeFile)(PyObject *_self, PyObject *args)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  PyObject *file = PyTuple_GetItem(args, 0);
  bool is_real_file = PyTuple_GetItem(args, 1) == Py_True;
  bool save_size = PyTuple_GetItem(args, 2) == Py_True;

  if (!is_real_file) {
    THPStorage_(writeFileRaw<PyObject*>)(self->cdata, file, save_size);
    Py_RETURN_NONE;
  }

  int fd = PyObject_AsFileDescriptor(file);
  THPUtils_assert(fd != -1, "_write_file couldn't retrieve a file descriptor "
      "from given object");
  THPStorage_(writeFileRaw)(self->cdata, fd, save_size);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THPStorage_(newWithFile)(PyObject *_unused, PyObject *file)
{
  HANDLE_TH_ERRORS
  int fd = PyObject_AsFileDescriptor(file);
  THPUtils_assert(fd != -1, "_new_with_file couldn't retrieve a file "
      "descriptor from given object");
  THWStorage *storage = THPStorage_(readFileRaw<int>)(fd, nullptr);
  if (storage == nullptr)
    return nullptr;
  PyObject *result = THPStorage_(New)(storage);
  return result;
  END_HANDLE_TH_ERRORS
}

static PyObject *THPStorage_(setFromFile)(PyObject *_self, PyObject *args)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  PyObject *file = PyTuple_GET_ITEM(args, 0);
  PyObject *offset = PyTuple_GET_ITEM(args, 1);
  bool is_real_file = PyTuple_GET_ITEM(args, 2) == Py_True;

  if (!is_real_file) {
    // offset can be implemented with a call to the Python object's seek()
    // but it is currently unnecessary to support this.
    THPUtils_assert(offset == Py_None,
                    "_set_from_file: offset is NYI for filelike objects");
    THWStorage *storage = THPStorage_(readFileRaw<PyObject*>)(file, self->cdata);
    if (storage == nullptr) {
      return nullptr;
    }
    Py_INCREF(self);
    return (PyObject *) self;
  }

  // file is backed by a fd
  const int fd = PyObject_AsFileDescriptor(file);
  const auto fd_original_pos = LSEEK(fd, 0, SEEK_CUR);
  if (offset != Py_None) {
    LSEEK(fd, THPUtils_unpackLong(offset), SEEK_SET);
  }
  THPUtils_assert(fd != -1, "_set_from_file couldn't retrieve a file "
      "descriptor from given object");
  THWStorage *storage = THPStorage_(readFileRaw<int>)(fd, self->cdata);
  if (storage == nullptr)
    return nullptr;
  Py_INCREF(self);

  // the file descriptor is returned to original position and
  // the file handle at python call-site needs updating to the
  // advanced position
  const auto fd_current_pos = LSEEK(fd, 0, SEEK_CUR);
  LSEEK(fd, fd_original_pos, SEEK_SET);
  const auto seek_return = PyObject_CallMethod(file, "seek", "Li", (long long)fd_current_pos, 0);
  if (seek_return == nullptr) {
      return nullptr;
  }
  Py_DECREF(seek_return);

  return (PyObject *) self;
  END_HANDLE_TH_ERRORS
}

#ifdef THC_GENERIC_FILE
PyObject * THPStorage_(getDevice)(PyObject *_self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  return THPUtils_packInt32(THCStorage_(getDevice)(LIBRARY_STATE self->cdata));
  END_HANDLE_TH_ERRORS
}
#endif

PyObject * THPStorage_(_setCdata)(PyObject *_self, PyObject *new_cdata)
{
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  THPUtils_assert(THPUtils_checkLong(new_cdata), "given an invalid argument to "
      "_set_cdata - expected an int or long, but got %s",
      THPUtils_typename(new_cdata));
  THWStorage *ptr = (THWStorage*)PyLong_AsVoidPtr(new_cdata);
  THWStorage_(retain)(LIBRARY_STATE ptr);
  THWStorage_(free)(LIBRARY_STATE self->cdata);
  self->cdata = ptr;
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef THPStorage_(methods)[] = {
  {"copy_", castPyCFunctionWithKeywords(THPStorage_(copy_)),
    METH_VARARGS | METH_KEYWORDS, nullptr},
  {"element_size", THPStorage_(elementSize), METH_NOARGS, nullptr},
  {"fill_", THPStorage_(fill_), METH_O, nullptr},
  {"new", THPStorage_(new), METH_NOARGS, nullptr},
  {"resize_", THPStorage_(resize_), METH_O, nullptr},
  {"size", THPStorage_(size), METH_NOARGS, nullptr},
  {"data_ptr", THPStorage_(dataPtr), METH_NOARGS, nullptr},
  {"is_pinned", THPStorage_(isPinned), METH_NOARGS, nullptr},
  {"_write_file", THPStorage_(writeFile), METH_VARARGS, nullptr},
  {"_new_with_file", THPStorage_(newWithFile), METH_O | METH_STATIC, nullptr},
  {"_set_from_file", THPStorage_(setFromFile), METH_VARARGS, nullptr},
#if !defined(THC_GENERIC_FILE)
  {"from_buffer", castPyCFunctionWithKeywords(THPStorage_(fromBuffer)),
    METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
#endif
  {"from_file", castPyCFunctionWithKeywords(THPStorage_(fromFile)),
    METH_VARARGS | METH_KEYWORDS | METH_STATIC, nullptr},
#ifdef THC_GENERIC_FILE
  {"get_device", THPStorage_(getDevice), METH_NOARGS, nullptr},
#endif
  {"_set_cdata", THPStorage_(_setCdata), METH_O, nullptr},
  {nullptr}
};
