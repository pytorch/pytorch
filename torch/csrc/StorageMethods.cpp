#include <torch/csrc/python_headers.h>
#ifdef _MSC_VER
#include <c10/util/win32-headers.h>
#endif
#include <structmember.h>

#include <c10/core/CPUAllocator.h>
#include <c10/util/overflows.h>
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
#include <torch/csrc/StorageMethods.h>

#include <ATen/ATen.h>
#include <ATen/MapAllocator.h>
#include <ATen/StorageUtils.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>

#ifdef USE_CUDA
#include <ATen/native/cuda/Resize.h>
#include <cuda_runtime.h>
#endif

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/Resize.h>

#ifdef _MSC_VER
#define LSEEK _lseeki64
#else
#define LSEEK lseek
#endif

static PyObject* THPStorage_nbytes(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  return py::cast(THPStorage_Unpack(self).sym_nbytes()).release().ptr();
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_dataPtr(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // PyLong_FromVoidPtr should not need to mutate the pointer in order
  // to extract a new long object from it.

  auto self_ = THPStorage_Unpack(self);
  // See Note [Invalid Python Storages]
  auto invalid = self_.data() == nullptr &&
      self_.device_type() != c10::DeviceType::Meta && self_.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid,
      "Attempted to access the data pointer on an invalid python storage.")
  return PyLong_FromVoidPtr(self_.mutable_data());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_resizable(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  return PyBool_FromLong(THPStorage_Unpack(self).resizable());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_copy_(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);

  at::Storage self_ = torch::createStorage(self);

  static torch::PythonArgParser parser({
      "copy_(Storage src, bool? non_blocking=None)",
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::Storage src = r.storage(0);
  bool non_blocking = r.toBoolOptional(1).value_or(false);

  // See Note [Invalid Python Storages]
  auto invalid = src.data() == nullptr &&
      src.device_type() != c10::DeviceType::Meta && src.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid, "Attempted to call copy_() on an invalid python storage.")

  TORCH_CHECK(
      self_.nbytes() == src.nbytes(),
      "size does not match, self was ",
      self_.nbytes(),
      " bytes but src was ",
      src.nbytes(),
      " bytes");

  at::storage_copy(self_, src, non_blocking);

  Py_INCREF(self);
  return self;

  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_elementSize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(_self);
  return THPUtils_packInt64(sizeof(uint8_t));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_new(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  c10::Allocator* allocator = THPStorage_Unpack(self).allocator();
  auto new_storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      0,
      allocator,
      /*resizable=*/true);

  return THPStorage_Wrap(std::move(new_storage));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_resize_(PyObject* self, PyObject* number_arg) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(self);
  // See Note [Invalid Python Storages]
  auto invalid = storage.data() == nullptr &&
      storage.device_type() != c10::DeviceType::Meta &&
      storage.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid, "Attempted to call resize_() on an invalid python storage.")
  TORCH_CHECK(
      THPUtils_checkLong(number_arg),
      "resize_ expects an int, "
      "but got ",
      THPUtils_typename(number_arg));
  int64_t newsize = THPUtils_unpackLong(number_arg);
  c10::DeviceType device_type = storage.device_type();
  if (device_type == at::kCUDA) {
#ifdef USE_CUDA
    ptrdiff_t size_bytes_i = newsize;
    TORCH_CHECK(
        !c10::overflows<size_t>(size_bytes_i),
        "Requested storage size (",
        size_bytes_i,
        ") cannot be represented as a size_t");
    const auto size_bytes = static_cast<size_t>(size_bytes_i);
    at::native::resize_bytes_cuda(storage.unsafeGetStorageImpl(), size_bytes);
#else
    TORCH_CHECK(false, "built without USE_CUDA");
#endif
  } else {
    at::native::resize_bytes_nocuda(storage, newsize);
  }
  Py_INCREF(self);
  return self;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_fill_(PyObject* self, PyObject* number_arg) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(self);
  // See Note [Invalid Python Storages]
  auto invalid = storage.data() == nullptr &&
      storage.device_type() != c10::DeviceType::Meta &&
      storage.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid, "Attempted to call fill_() on an invalid python storage.")
  TORCH_CHECK(
      THPByteUtils_checkReal(number_arg),
      "fill_ expects int, "
      "but got ",
      THPUtils_typename(number_arg));
  storage_fill(storage, THPByteUtils_unpackReal(number_arg));
  Py_INCREF(self);
  return self;
  END_HANDLE_TH_ERRORS
}

template <typename T>
static void decodeWrapper(
    void* data,
    const uint8_t* src,
    bool do_byte_swap,
    size_t count) {
  torch::utils::THP_decodeBuffer(
      static_cast<T*>(data), src, do_byte_swap, count);
}

static PyObject* THPStorage_fromBuffer(
    PyObject* _unused,
    PyObject* args,
    PyObject* keywds) {
  HANDLE_TH_ERRORS
  PyObject* obj = nullptr;
  const char* byte_order_str = nullptr;
  Py_ssize_t count = -1, offset = 0;
  PyObject* dtype_obj = nullptr;
  c10::ScalarType scalar_type = at::kByte;
  Py_buffer buffer = {};
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "buffer", "byte_order", "count", "offset", "dtype", nullptr};
  constexpr const char* argtypes = "O|snnO";

  if (!PyArg_ParseTupleAndKeywords(
          args,
          keywds,
          argtypes,
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &obj,
          &byte_order_str,
          &count,
          &offset,
          &dtype_obj)) {
    return nullptr;
  }
  TORCH_CHECK(dtype_obj != nullptr, "argument 'dtype' cannot be None");
  TORCH_CHECK(
      THPDtype_Check(dtype_obj),
      "argument 'dtype' must be of type torch.dtype");
  auto dtype = reinterpret_cast<THPDtype*>(dtype_obj);
  scalar_type = dtype->scalar_type;

  const bool is_endian_independent = (scalar_type == at::kByte) ||
      (scalar_type == at::kChar) || (scalar_type == at::kFloat8_e5m2) ||
      (scalar_type == at::kFloat8_e5m2fnuz) ||
      (scalar_type == at::kFloat8_e4m3fn) ||
      (scalar_type == at::kFloat8_e4m3fnuz);

  TORCH_CHECK(
      is_endian_independent || (byte_order_str != nullptr),
      "function missing required argument 'byte_order' (pos 2)");
  size_t element_size = c10::elementSize(scalar_type);

  bool do_byte_swap = false;
  if (!is_endian_independent) {
    if (strcmp(byte_order_str, "native") == 0) {
      do_byte_swap = false;
    } else if (strcmp(byte_order_str, "big") == 0) {
      do_byte_swap =
          (torch::utils::THP_LITTLE_ENDIAN ==
           torch::utils::THP_nativeByteOrder());
    } else if (strcmp(byte_order_str, "little") == 0) {
      do_byte_swap =
          (torch::utils::THP_BIG_ENDIAN == torch::utils::THP_nativeByteOrder());
    } else {
      PyErr_Format(
          PyExc_ValueError,
          "invalid byte_order '%s' (expected 'big', 'little', or 'native')",
          byte_order_str);
      return nullptr;
    }
  }

  if (PyObject_GetBuffer(obj, &buffer, PyBUF_SIMPLE) < 0)
    return nullptr;

  if (offset < 0 || offset > buffer.len) {
    PyErr_SetString(
        PyExc_ValueError,
        fmt::format(
            "offset must be non-negative and no greater than buffer length ({}) , but got {}",
            offset,
            buffer.len));
    PyBuffer_Release(&buffer);
    return nullptr;
  }

  size_t size_bytes = 0;
  if (count < 0) {
    if ((buffer.len - offset) % element_size != 0) {
      PyErr_SetString(
          PyExc_ValueError,
          fmt::format(
              "buffer size ({}) must be a multiple of element size ({})",
              buffer.len,
              element_size));
      PyBuffer_Release(&buffer);
      return nullptr;
    }
    size_bytes = buffer.len - offset;
    count = static_cast<Py_ssize_t>(size_bytes / element_size);
  } else {
    size_bytes = count * element_size;
  }

  if (offset + (count * (Py_ssize_t)element_size) > buffer.len) {
    PyErr_SetString(
        PyExc_ValueError,
        fmt::format(
            "buffer has only {} elements after offset {}, but specified a size of {}",
            buffer.len - offset,
            offset,
            count));
    PyBuffer_Release(&buffer);
    return nullptr;
  }

  uint8_t* src = (uint8_t*)buffer.buf;
  auto storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      c10::GetDefaultCPUAllocator(),
      /*resizable=*/true);

  static const std::unordered_map<
      at::ScalarType,
      std::function<void(void*, const uint8_t*, bool, size_t)>>
      decode_map = {
          {at::kBool, decodeWrapper<bool>},
          {at::kShort, decodeWrapper<int16_t>},
          {at::kInt, decodeWrapper<int32_t>},
          {at::kLong, decodeWrapper<int64_t>},
          {at::kHalf, decodeWrapper<c10::Half>},
          {at::kBFloat16, decodeWrapper<c10::BFloat16>},
          {at::kFloat, decodeWrapper<float>},
          {at::kDouble, decodeWrapper<double>},
          {at::kComplexFloat, decodeWrapper<c10::complex<float>>},
          {at::kComplexDouble, decodeWrapper<c10::complex<double>>}};

  if (is_endian_independent) {
    memcpy(storage->mutable_data(), src + offset, count);
  } else {
    auto it = decode_map.find(scalar_type);
    if (it != decode_map.end()) {
      it->second(storage->mutable_data(), src + offset, do_byte_swap, count);
    } else {
      TORCH_CHECK(false, "Unknown type: ", scalar_type);
    }
  }

  PyBuffer_Release(&buffer);
  return THPStorage_Wrap(storage);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_fromFile(
    PyObject* _unused,
    PyObject* args,
    PyObject* keywds) {
  HANDLE_TH_ERRORS
  const char* filename = nullptr;
  Py_ssize_t nbytes = 0;
  int shared = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* kwlist[] = {"filename", "shared", "nbytes", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          keywds,
          "s|in",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &filename,
          &shared,
          &nbytes)) {
    return nullptr;
  }
  if (shared)
    shared = at::ALLOCATOR_MAPPED_SHARED;

  size_t actual_nbytes = -1;
  auto storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      nbytes,
      at::MapAllocator::makeDataPtr(filename, shared, nbytes, &actual_nbytes),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  if (nbytes <= 0) {
    storage->set_nbytes(actual_nbytes);
  }

  return THPStorage_NewWithStorage(
      THPStorageClass,
      std::move(storage),
      c10::impl::PyInterpreterStatus::TAGGED_BY_US);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_writeFile(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(self);
  // See Note [Invalid Python Storages]
  auto invalid = storage.data() == nullptr &&
      storage.device_type() != c10::DeviceType::Meta &&
      storage.sym_nbytes() != 0;
  TORCH_CHECK(
      !invalid, "Attempted to call _write_file() on an invalid python storage.")
  PyObject* file = PyTuple_GetItem(args, 0);
  bool is_real_file = PyTuple_GetItem(args, 1) == Py_True;
  bool save_size = PyTuple_GetItem(args, 2) == Py_True;
  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 3);

  TORCH_CHECK(
      element_size_obj != Py_None, "_write_file: need to specify element size");
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);

  if (!is_real_file) {
    THPStorage_writeFileRaw<PyObject*>(
        storage.unsafeGetStorageImpl(), file, save_size, element_size);
    Py_RETURN_NONE;
  }

  int fd = PyObject_AsFileDescriptor(file);
  TORCH_CHECK(
      fd != -1,
      "_write_file couldn't retrieve a file descriptor "
      "from given object");
  THPStorage_writeFileRaw(
      storage.unsafeGetStorageImpl(), fd, save_size, element_size);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_newWithFile(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyTuple_Size(args) == 2, "_new_with_file takes exactly two arguments");
  int fd = PyObject_AsFileDescriptor(PyTuple_GetItem(args, 0));
  TORCH_CHECK(
      fd != -1,
      "_new_with_file couldn't retrieve a file "
      "descriptor from given object");
  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 1);
  TORCH_CHECK(
      element_size_obj != Py_None,
      "_new_with_file: need to specify element size");
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);

  auto storage = THPStorage_readFileRaw<int>(fd, {}, element_size);
  if (!storage.defined())
    return nullptr;
  return THPStorage_Wrap(std::move(storage));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_setFromFile(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  THPStorage_assertNotNull(self);
  const auto& storage = THPStorage_Unpack(self);
  PyObject* file = PyTuple_GET_ITEM(args, 0);
  PyObject* offset = PyTuple_GET_ITEM(args, 1);
  bool is_real_file = PyTuple_GET_ITEM(args, 2) == Py_True;

  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 3);

  TORCH_CHECK(
      element_size_obj != Py_None,
      "_set_from_file: need to specify element size");
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);

  if (!is_real_file) {
    // offset can be implemented with a call to the Python object's seek()
    // but it is currently unnecessary to support this.
    TORCH_CHECK(
        offset == Py_None,
        "_set_from_file: offset is NYI for filelike objects");

    auto self_storage_impl = c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(
        storage.unsafeGetStorageImpl());
    auto storage_impl = THPStorage_readFileRaw<PyObject*>(
        file, std::move(self_storage_impl), element_size);
    if (!storage_impl.defined()) {
      return nullptr;
    }
    Py_INCREF(self);
    return (PyObject*)self;
  }

  // file is backed by a fd
  const int fd = PyObject_AsFileDescriptor(file);
  const auto fd_original_pos = LSEEK(fd, 0, SEEK_CUR);
  if (offset != Py_None) {
    LSEEK(fd, THPUtils_unpackLong(offset), SEEK_SET);
  }
  TORCH_CHECK(
      fd != -1,
      "_set_from_file couldn't retrieve a file "
      "descriptor from given object");
  auto self_storage_impl = c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(
      storage.unsafeGetStorageImpl());
  auto storage_impl =
      THPStorage_readFileRaw<int>(fd, self_storage_impl, element_size);
  if (!storage_impl.defined())
    return nullptr;
  Py_INCREF(self);

  // the file descriptor is returned to original position and
  // the file handle at python call-site needs updating to the
  // advanced position
  const auto fd_current_pos = LSEEK(fd, 0, SEEK_CUR);
  LSEEK(fd, fd_original_pos, SEEK_SET);
  const auto seek_return =
      PyObject_CallMethod(file, "seek", "Li", (long long)fd_current_pos, 0);
  if (seek_return == nullptr) {
    return nullptr;
  }
  Py_DECREF(seek_return);

  return self;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage__setCdata(PyObject* _self, PyObject* new_cdata) {
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  TORCH_CHECK(
      THPUtils_checkLong(new_cdata),
      "given an invalid argument to "
      "_set_cdata - expected an int or long, but got ",
      THPUtils_typename(new_cdata));
  c10::StorageImpl* ptr = (c10::StorageImpl*)PyLong_AsVoidPtr(new_cdata);
  self->cdata.~MaybeOwned<c10::Storage>();
  self->cdata = c10::MaybeOwned<c10::Storage>::owned(
      c10::Storage(c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(ptr)));
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_byteswap(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 1, "tuple of 1 item expected");
  PyObject* _elem_size = PyTuple_GET_ITEM(args, 0);
  TORCH_CHECK(
      THPUtils_checkLong(_elem_size), "_byteswap(): arg must be an 'int'");
  auto elem_size = THPUtils_unpackLong(_elem_size);
  TORCH_CHECK(
      elem_size == 1 || elem_size == 2 || elem_size == 4 || elem_size == 8,
      "elem_size must be 1, 2, 4, or 8");

  const auto& storage = THPStorage_Unpack(self);
  const auto nbytes = static_cast<uint64_t>(storage.nbytes());
  const uint64_t count = nbytes / elem_size;

  if (elem_size == 1) {
    Py_RETURN_NONE;
  }
  TORCH_CHECK(
      nbytes % elem_size == 0,
      "the length of data is not a multiple of ",
      elem_size);

  if (elem_size == 2) {
    auto buffer = static_cast<uint16_t*>(storage.mutable_data());
    for (uint64_t i = 0; i < count; i++, buffer++) {
      *buffer = thp_bswap16(*buffer);
    }
  } else if (elem_size == 4) {
    auto buffer = static_cast<uint32_t*>(storage.mutable_data());
    for (uint64_t i = 0; i < count; i++, buffer++) {
      *buffer = thp_bswap32(*buffer);
    }
  } else if (elem_size == 8) {
    auto buffer = static_cast<uint64_t*>(storage.mutable_data());
    for (uint64_t i = 0; i < count; i++, buffer++) {
      *buffer = thp_bswap64(*buffer);
    }
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_fix_weakref(PyObject* self, PyObject* noargs) {
  const auto& storage = THPStorage_Unpack(self);
  Py_DECREF(THPStorage_Wrap(storage));
  Py_RETURN_NONE;
}

static PyObject* THPStorage__get_filename(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  const auto& self_ = THPStorage_Unpack(self);
  const c10::DataPtr& data_ptr = self_.data_ptr();
  at::MapAllocator* map_allocator = at::MapAllocator::fromDataPtr(data_ptr);

  if (map_allocator == nullptr ||
      !(map_allocator->flags() & at::ALLOCATOR_MAPPED_SHARED)) {
    Py_RETURN_NONE;
  }
  std::string filename = map_allocator->filename();

  return THPUtils_packString(filename);
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef THPStorage_methods[] = {
    {"copy_",
     castPyCFunctionWithKeywords(THPStorage_copy_),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"element_size", THPStorage_elementSize, METH_NOARGS, nullptr},
    {"fill_", THPStorage_fill_, METH_O, nullptr},
    {"new", THPStorage_new, METH_NOARGS, nullptr},
    {"resize_", THPStorage_resize_, METH_O, nullptr},
    {"nbytes", THPStorage_nbytes, METH_NOARGS, nullptr},
    {"data_ptr", THPStorage_dataPtr, METH_NOARGS, nullptr},
    {"resizable", THPStorage_resizable, METH_NOARGS, nullptr},
    {"_write_file", THPStorage_writeFile, METH_VARARGS, nullptr},
    {"_new_with_file",
     THPStorage_newWithFile,
     METH_VARARGS | METH_STATIC,
     nullptr},
    {"_set_from_file", THPStorage_setFromFile, METH_VARARGS, nullptr},
    {"from_buffer",
     castPyCFunctionWithKeywords(THPStorage_fromBuffer),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"from_file",
     castPyCFunctionWithKeywords(THPStorage_fromFile),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"_set_cdata", THPStorage__setCdata, METH_O, nullptr},
    {"_byteswap", THPStorage_byteswap, METH_VARARGS, nullptr},
    {"_fix_weakref", THPStorage_fix_weakref, METH_NOARGS, nullptr},
    {"_get_filename", THPStorage__get_filename, METH_NOARGS, nullptr},
    {nullptr}};

PyMethodDef* THPStorage_getMethods() {
  return THPStorage_methods;
}
