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

#include <ATen/native/Resize.h>

#ifdef _MSC_VER
#define LSEEK _lseeki64
#else
#define LSEEK lseek
#endif

static PyObject* THPStorage_nbytes(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return py::cast(THPStorage_Unpack(self).sym_nbytes()).release().ptr();
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_dataPtr(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // PyLong_FromVoidPtr should not need to mutate the pointer in order
  // to extract a new long object from it.
  return PyLong_FromVoidPtr(const_cast<void*>(THPStorage_Unpack(self).data()));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_copy_(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  at::Storage self_ = torch::createStorage(self);

  static torch::PythonArgParser parser({
      "copy_(Storage src, bool? non_blocking=None)",
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::Storage src = r.storage(0);
  bool non_blocking = r.toBoolOptional(1).value_or(false);

  TORCH_CHECK(self_.nbytes() == src.nbytes(), "size does not match");

  at::storage_copy(self_, src, non_blocking);

  Py_INCREF(self);
  return self;

  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_elementSize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(sizeof(uint8_t));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_new(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::Allocator* allocator = THPStorage_Unpack(self).allocator();
  auto new_storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      0,
      allocator,
      /*resizable=*/true);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  return THPStorage_New(std::move(new_storage));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_resize_(PyObject* self, PyObject* number_arg) {
  HANDLE_TH_ERRORS
  const auto& storage = THPStorage_Unpack(self);
  THPUtils_assert(
      THPUtils_checkLong(number_arg),
      "resize_ expects an int, "
      "but got %s",
      THPUtils_typename(number_arg));
  int64_t newsize = THPUtils_unpackLong(number_arg);
  c10::DeviceType device_type = storage.device_type();
  if (device_type == at::kCPU) {
    at::native::resize_bytes_cpu(storage.unsafeGetStorageImpl(), newsize);
#ifdef USE_CUDA
  } else if (device_type == at::kCUDA) {
    ptrdiff_t size_bytes_i = newsize;
    TORCH_CHECK(
        !c10::overflows<size_t>(size_bytes_i),
        "Requested storage size (",
        size_bytes_i,
        ") cannot be represented as a size_t");
    const auto size_bytes = static_cast<size_t>(size_bytes_i);
    at::native::resize_bytes_cuda(storage.unsafeGetStorageImpl(), size_bytes);
#endif
  } else if (device_type == at::kMeta) {
    at::native::resize_bytes_meta(storage.unsafeGetStorageImpl(), newsize);
  } else if (device_type == at::kPrivateUse1) {
    ptrdiff_t size_bytes_i = newsize;
    TORCH_CHECK(
        !c10::overflows<int64_t>(size_bytes_i),
        "Requested storage size (",
        size_bytes_i,
        ") cannot be represented as a int64_t");
    const auto size_bytes = static_cast<int64_t>(size_bytes_i);
    void* original_data_ptr = storage.data_ptr().get();

    auto src_option =
        c10::TensorOptions().device(storage.device()).dtype(at::kByte);
    auto src_tensor = at::empty({0}, {}, src_option).set_(storage);
    src_tensor.resize_({size_bytes});

    // When using resize_ to replace resize_bytes_xxx, in some cases
    // the original data_ptr is still returned, which is an inconsistent
    // behavior when compared to resize_bytes_xxx. For these cases,
    // an additional memory copy and update for storage are required.
    if (original_data_ptr == src_tensor.storage().data_ptr().get()) {
      auto new_tensor = at::empty(src_tensor.sizes(), src_tensor.options());
      new_tensor.copy_(src_tensor);
      storage.set_data_ptr_noswap(
          std::move(const_cast<at::DataPtr&>(new_tensor.storage().data_ptr())));
      storage.unsafeGetStorageImpl()->set_allocator(
          new_tensor.storage().unsafeGetStorageImpl()->allocator());
      storage.set_nbytes(new_tensor.storage().nbytes());
    }
  } else {
    TORCH_CHECK(
        false,
        "UntypedStorage.resize_: got unexpected device type ",
        device_type);
  }
  Py_INCREF(self);
  return self;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_fill_(PyObject* self, PyObject* number_arg) {
  HANDLE_TH_ERRORS
  const auto& storage = THPStorage_Unpack(self);
  THPUtils_assert(
      THPByteUtils_checkReal(number_arg),
      "fill_ expects int, "
      "but got %s",
      THPUtils_typename(number_arg));
  storage_fill(storage, THPByteUtils_unpackReal(number_arg));
  Py_INCREF(self);
  return self;
  END_HANDLE_TH_ERRORS
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

  TORCH_CHECK(
      (scalar_type == at::kByte) || (scalar_type == at::kChar) ||
          (byte_order_str != nullptr),
      "function missing required argument 'byte_order' (pos 2)");
  size_t element_size = c10::elementSize(scalar_type);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool do_byte_swap;
  if (scalar_type != at::kByte && scalar_type != at::kChar) {
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

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t size_bytes;
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
    count = size_bytes / element_size;
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

  if (scalar_type == at::kByte || scalar_type == at::kChar) {
    memcpy(storage->mutable_data(), src + offset, count);
  } else if (scalar_type == at::kBool) {
    // Because of ASAN checks, that are failing whenever
    // we are trying to get a value which is not 0 or 1, we have to manually
    // convert original values to boolean ones.
    torch::utils::THP_decodeBoolBuffer(
        static_cast<bool*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kShort) {
    torch::utils::THP_decodeInt16Buffer(
        static_cast<int16_t*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kInt) {
    torch::utils::THP_decodeInt32Buffer(
        static_cast<int32_t*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kLong) {
    torch::utils::THP_decodeInt64Buffer(
        static_cast<int64_t*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kHalf) {
    torch::utils::THP_decodeHalfBuffer(
        static_cast<c10::Half*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kBFloat16) {
    torch::utils::THP_decodeBFloat16Buffer(
        static_cast<c10::BFloat16*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kFloat) {
    torch::utils::THP_decodeFloatBuffer(
        static_cast<float*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kDouble) {
    torch::utils::THP_decodeDoubleBuffer(
        static_cast<double*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kComplexFloat) {
    torch::utils::THP_decodeComplexFloatBuffer(
        static_cast<c10::complex<float>*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else if (scalar_type == at::kComplexDouble) {
    torch::utils::THP_decodeComplexDoubleBuffer(
        static_cast<c10::complex<double>*>(storage->mutable_data()),
        src + offset,
        do_byte_swap,
        count);
  } else {
    TORCH_CHECK(false, "Unknown type: ", scalar_type);
  }

  PyBuffer_Release(&buffer);
  return (PyObject*)THPStorage_New(storage);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_fromFile(
    PyObject* _unused,
    PyObject* args,
    PyObject* keywds) {
  HANDLE_TH_ERRORS
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const char* filename;
  Py_ssize_t nbytes = 0;
  int shared = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* kwlist[] = {"filename", "shared", "nbytes", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          keywds,
          "s|in",
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

  return (PyObject*)THPStorage_New(std::move(storage));
  END_HANDLE_TH_ERRORS
}

PyObject* THPStorage_writeFile(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  const auto& storage = THPStorage_Unpack(self);
  PyObject* file = PyTuple_GetItem(args, 0);
  bool is_real_file = PyTuple_GetItem(args, 1) == Py_True;
  bool save_size = PyTuple_GetItem(args, 2) == Py_True;
  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 3);

  THPUtils_assert(
      element_size_obj != Py_None, "_write_file: need to specify element size");
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);

  if (!is_real_file) {
    THPStorage_writeFileRaw<PyObject*>(
        storage.unsafeGetStorageImpl(), file, save_size, element_size);
    Py_RETURN_NONE;
  }

  int fd = PyObject_AsFileDescriptor(file);
  THPUtils_assert(
      fd != -1,
      "_write_file couldn't retrieve a file descriptor "
      "from given object");
  THPStorage_writeFileRaw(
      storage.unsafeGetStorageImpl(), fd, save_size, element_size);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPStorage_newWithFile(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyTuple_Size(args) == 2, "_new_with_file takes exactly two arguments");
  int fd = PyObject_AsFileDescriptor(PyTuple_GetItem(args, 0));
  THPUtils_assert(
      fd != -1,
      "_new_with_file couldn't retrieve a file "
      "descriptor from given object");
  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 1);
  THPUtils_assert(
      element_size_obj != Py_None,
      "_new_with_file: need to specify element size");
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);

  auto storage = THPStorage_readFileRaw<int>(fd, {}, element_size);
  if (!storage.defined())
    return nullptr;
  return THPStorage_New(std::move(storage));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPStorage_setFromFile(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  const auto& storage = THPStorage_Unpack(self);
  PyObject* file = PyTuple_GET_ITEM(args, 0);
  PyObject* offset = PyTuple_GET_ITEM(args, 1);
  bool is_real_file = PyTuple_GET_ITEM(args, 2) == Py_True;

  PyObject* element_size_obj = PyTuple_GET_ITEM(args, 3);

  THPUtils_assert(
      element_size_obj != Py_None,
      "_set_from_file: need to specify element size");
  uint64_t element_size = THPUtils_unpackUInt64(element_size_obj);

  if (!is_real_file) {
    // offset can be implemented with a call to the Python object's seek()
    // but it is currently unnecessary to support this.
    THPUtils_assert(
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
  THPUtils_assert(
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

PyObject* THPStorage__setCdata(PyObject* _self, PyObject* new_cdata) {
  HANDLE_TH_ERRORS
  auto self = (THPStorage*)_self;
  THPUtils_assert(
      THPUtils_checkLong(new_cdata),
      "given an invalid argument to "
      "_set_cdata - expected an int or long, but got %s",
      THPUtils_typename(new_cdata));
  c10::StorageImpl* ptr = (c10::StorageImpl*)PyLong_AsVoidPtr(new_cdata);
  self->cdata.~MaybeOwned<c10::Storage>();
  self->cdata = c10::MaybeOwned<c10::Storage>::owned(
      c10::Storage(c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(ptr)));
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

PyObject* THPStorage_byteswap(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  THPUtils_assert(PyTuple_GET_SIZE(args) == 1, "tuple of 1 item expected");
  PyObject* _elem_size = PyTuple_GET_ITEM(args, 0);
  THPUtils_assert(
      THPUtils_checkLong(_elem_size), "_byteswap(): arg must be an 'int'");
  auto elem_size = THPUtils_unpackLong(_elem_size);
  THPUtils_assert(
      elem_size == 1 || elem_size == 2 || elem_size == 4 || elem_size == 8,
      "elem_size must be 1, 2, 4, or 8");

  const auto& storage = THPStorage_Unpack(self);
  const auto nbytes = static_cast<uint64_t>(storage.nbytes());
  const uint64_t count = nbytes / elem_size;

  if (elem_size == 1) {
    Py_RETURN_NONE;
  }
  THPUtils_assert(
      nbytes % elem_size == 0,
      "the length of data is not a multiple of %ld",
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
    {nullptr}};

PyMethodDef* THPStorage_getMethods() {
  return THPStorage_methods;
}
