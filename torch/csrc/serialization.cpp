#include <torch/csrc/python_headers.h>
#include <system_error>

#include <ATen/ops/from_blob.h>
#include <c10/core/CPUAllocator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/serialization.h>

template <class io>
Py_ssize_t doPartialRead(io fildes, void* buf, size_t nbytes);

template <class io>
Py_ssize_t doPartialWrite(io fildes, void* buf, size_t nbytes);

static Py_ssize_t doPartialPythonReadBuffered(
    PyObject* fildes,
    void* buf,
    size_t nbytes);
static Py_ssize_t doPartialPythonReadInto(
    PyObject* fildes,
    void* buf,
    size_t nbytes);
static Py_ssize_t doPartialPythonWrite(
    PyObject* fildes,
    void* buf,
    size_t nbytes);

template <>
Py_ssize_t doPartialRead<int>(int fildes, void* buf, size_t nbytes) {
  return read(fildes, buf, nbytes);
}

template <>
Py_ssize_t doPartialRead<PyObject*>(
    PyObject* fildes,
    void* buf,
    size_t nbytes) {
  // Try to use fildes.readinto() instead of fildes.read()
  // because it is more memory efficient.
  // TODO: Stop calling PyObject_HasAttrString() in a loop on our read loop
  auto has_readinto = PyObject_HasAttrString(fildes, "readinto") == 1;
  if (has_readinto) {
    return doPartialPythonReadInto(fildes, buf, nbytes);
  }
  return doPartialPythonReadBuffered(fildes, buf, nbytes);
}

template <>
Py_ssize_t doPartialWrite<int>(int fildes, void* buf, size_t nbytes) {
  return write(fildes, buf, nbytes);
}

template <>
Py_ssize_t doPartialWrite<PyObject*>(
    PyObject* fildes,
    void* buf,
    size_t nbytes) {
  return doPartialPythonWrite(fildes, buf, nbytes);
}

static inline bool isUnsupportedOperation() {
  THPObjectPtr io(PyImport_ImportModule("io"));
  if (!io)
    throw python_error();
  THPObjectPtr exception(PyObject_GetAttrString(io, "UnsupportedOperation"));
  if (!exception)
    throw python_error();
  return PyErr_ExceptionMatches(exception.get());
}

// Call Python fildes.read(nbytes) and copy it to buf.
static inline Py_ssize_t doPartialPythonReadBuffered(
    PyObject* fildes,
    void* buf,
    size_t raw_nbytes) {
  // If we request a large amount of data, f.read() will internally try to
  // allocate a buffer of that size.  This is counterproductive, because
  // it's not the buffer we ultimately want to write the data into.  Read
  // less than that and avoid allocating too much extra memory.
  // TODO: Maybe 260 KB is a bit small...
  const size_t nbytes = std::min<size_t>(raw_nbytes, 262144u); // 2^18 (~260 KB)

  THPObjectPtr r(PyObject_CallMethod(fildes, "read", "i", nbytes));
  if (!r)
    throw python_error();

  auto size = PyBytes_GET_SIZE(r.get());
  const void* py_buf = PyBytes_AsString(r.get());

  // we read EOF
  if (size == 0) {
    return 0;
  }

  // Slurp it into the buffer we actually want
  memcpy(buf, py_buf, size);

  return size;
}

// Either does fildes.readinto(buf) or fildes.write(buf)
static inline Py_ssize_t doPartialPythonIO(
    PyObject* fildes,
    void* buf,
    size_t nbytes,
    bool is_read) {
  auto rw_flag = is_read ? PyBUF_WRITE : PyBUF_READ;
  THPObjectPtr memview(PyMemoryView_FromMemory(
      reinterpret_cast<char*>(buf), static_cast<Py_ssize_t>(nbytes), rw_flag));
  if (!memview)
    throw python_error();

  std::string method = "write";
  if (is_read) {
    method = "readinto";
  }
  THPObjectPtr r(
      PyObject_CallMethod(fildes, method.c_str(), "O", memview.get()));
  if (r) {
    return PyLong_AsSsize_t(r.get());
  }

  // fildes.readinto can return UnsupportedOperation so fall back to
  // fildes.read.
  if (is_read && isUnsupportedOperation()) {
    PyErr_Clear();
    return doPartialPythonReadBuffered(fildes, buf, nbytes);
  }
  throw python_error();
}

// Call Python fildes.readinto(buf)
static Py_ssize_t doPartialPythonReadInto(
    PyObject* fildes,
    void* buf,
    size_t nbytes) {
  return doPartialPythonIO(fildes, buf, nbytes, /* is_read */ true);
}

// Call Python fildes.write(buf)
static Py_ssize_t doPartialPythonWrite(
    PyObject* fildes,
    void* buf,
    size_t nbytes) {
  return doPartialPythonIO(fildes, buf, nbytes, /* is_read */ false);
}

// Requires that we read EXACTLY nbytes; fails if we don't.
template <typename io>
void doRead(io fildes, void* raw_buf, size_t nbytes) {
  char* buf = static_cast<char*>(raw_buf);
  while (nbytes > 0) {
    errno = 0; // doPartialRead may not set errno
    // we read in 1GB blocks to avoid bugs on Mac OS X Lion
    // see https://github.com/pytorch/pytorch/issues/1031 for more details
    Py_ssize_t r =
        doPartialRead(fildes, buf, std::min<size_t>(nbytes, 1073741824));
    if (r < 0) {
      int err = errno;
      TORCH_INTERNAL_ASSERT(
          err != 0, "read(): impossible! r < 0, but no errno was set");
      TORCH_INTERNAL_ASSERT(
          err != EAGAIN,
          "read(): non-blocking fd ",
          fildes,
          " read EAGAIN; cowardly refusing to spin-wait");
      if (err == EINTR) {
        continue;
      } else {
        AT_ERROR("read(): fd ", fildes, " failed with ", strerror(err));
      }
    } else if (r == 0) {
      break;
    }
    buf += r;
    // This is guaranteed by POSIX, but I just want to be double-sure
    // to not underflow a signed integer.
    AT_ASSERT(static_cast<size_t>(r) <= nbytes);
    nbytes -= r;
  }
  if (nbytes != 0) {
    AT_ERROR(
        "unexpected EOF, expected ",
        nbytes,
        " more bytes. The file might be corrupted.");
  }
}

template <typename io>
void doWrite(io fildes, void* raw_buf, size_t nbytes) {
  char* buf = static_cast<char*>(raw_buf);
  while (nbytes > 0) {
    errno = 0; // doPartialWrite may not set errno
    // we write in 1GB blocks to avoid bugs on Mac OS X Lion
    // see https://github.com/pytorch/pytorch/issues/1031 for more details
    Py_ssize_t r =
        doPartialWrite(fildes, buf, std::min<size_t>(nbytes, 1073741824));
    if (r < 0) {
      int err = errno;
      TORCH_INTERNAL_ASSERT(
          err != 0, "write(): impossible! r < 0, but no errno was set");
      TORCH_INTERNAL_ASSERT(
          err != EAGAIN,
          "write(): non-blocking fd ",
          fildes,
          " read EAGAIN; cowardly refusing to spin-wait");
      if (err == EINTR) {
        continue;
      } else {
        AT_ERROR("write(): fd ", fildes, " failed with ", strerror(err));
      }
    }
    buf += r;
    AT_ASSERT(static_cast<size_t>(r) <= nbytes);
    nbytes -= r;
  }
}

// save_save is necessary since the old eager format saved storages as
// [size + data], but the v1.5 eager format removes this since size is saved in
// the filesize.
template <class io>
void THPStorage_writeFileRaw(
    c10::StorageImpl* self,
    io fd,
    bool save_size,
    uint64_t element_size) {
  c10::DeviceGuard guard(self->device());
  uint8_t* data{};
  at::Tensor cpu_tensor;
  size_t size_bytes = self->nbytes();
  size_t numel = size_bytes / element_size;
  if (self->device_type() == at::kCPU) {
    // We are using a mutable pointer here because we're ultimately
    // calling into a Python API that requires that, even though it
    // won't mutate the data.
    data = static_cast<uint8_t*>(self->mutable_data());
  } else {
    // Here we use a tensor.to() to impl D2H for all non-CPU device.
    auto device_tensor = at::from_blob(
        self->mutable_data(),
        {static_cast<int64_t>(size_bytes)},
        {1},
        nullptr,
        at::device(self->device()).dtype(c10::kByte),
        {self->device()});
    cpu_tensor = device_tensor.to(at::kCPU);
    data = (uint8_t*)cpu_tensor.data_ptr();
  }
  if (save_size) {
    if (torch::utils::THP_nativeByteOrder() ==
        torch::utils::THPByteOrder::THP_LITTLE_ENDIAN)
      doWrite(fd, &numel, sizeof(int64_t));
    else {
      int64_t nsize{}; // convert big endian cpu to little endian storage
      torch::utils::THP_encodeInt64Buffer(
          (uint8_t*)&nsize,
          (const int64_t*)&numel,
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
          1);
      doWrite(fd, &nsize, sizeof(int64_t));
    }
  }
  // fast track for bytes and little endian
  if (element_size == 1 ||
      torch::utils::THP_nativeByteOrder() ==
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
    doWrite(fd, data, size_bytes);
  } else {
    size_t buffer_size = std::min(numel, (size_t)5000);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    std::unique_ptr<uint8_t[]> le_buffer(
        new uint8_t[buffer_size * element_size]);
    for (size_t i = 0; i < numel; i += buffer_size) {
      size_t to_convert = std::min(numel - i, buffer_size);
      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (element_size == 2) {
        torch::utils::THP_encodeInt16Buffer(
            (uint8_t*)le_buffer.get(),
            (const int16_t*)data + i,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (element_size == 4) {
        torch::utils::THP_encodeInt32Buffer(
            (uint8_t*)le_buffer.get(),
            (const int32_t*)data + i,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (element_size == 8) {
        torch::utils::THP_encodeInt64Buffer(
            (uint8_t*)le_buffer.get(),
            (const int64_t*)data + i,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      }
      doWrite(fd, le_buffer.get(), to_convert * element_size);
    }
  }
}

template void THPStorage_writeFileRaw<int>(
    c10::StorageImpl* self,
    int fd,
    bool save_size,
    uint64_t element_size);
template void THPStorage_writeFileRaw<PyObject*>(
    c10::StorageImpl* self,
    PyObject* fd,
    bool save_size,
    uint64_t element_size);

template <class io>
c10::intrusive_ptr<c10::StorageImpl> THPStorage_readFileRaw(
    io file,
    c10::intrusive_ptr<c10::StorageImpl> storage,
    uint64_t element_size) {
  c10::OptionalDeviceGuard guard;
  if (storage.defined()) {
    guard.reset_device(storage->device());
  }
  int64_t size{};
  doRead(file, &size, sizeof(int64_t));
  if (torch::utils::THP_nativeByteOrder() ==
      torch::utils::THPByteOrder::THP_BIG_ENDIAN) {
    int64_t tsize = size; // convert little endian storage to big endian cpu
    torch::utils::THP_decodeInt64Buffer(&size, (const uint8_t*)&tsize, true, 1);
  }
  size_t nbytes = element_size * size;
  if (!storage.defined()) {
    storage = c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        c10::GetDefaultCPUAllocator(),
        /*resizable=*/true);
  } else {
    size_t _storage_nbytes = storage->nbytes();
    TORCH_CHECK(
        _storage_nbytes == nbytes,
        "storage has wrong byte size: expected %ld got %ld",
        nbytes,
        _storage_nbytes);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  std::unique_ptr<char[]> cpu_data;

  uint8_t* data{};
  if (storage->device_type() == at::kCPU) {
    data = static_cast<uint8_t*>(storage->mutable_data());
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    cpu_data = std::unique_ptr<char[]>(new char[nbytes]);
    data = (uint8_t*)cpu_data.get();
  }

  // fast track for bytes and little endian
  if (element_size == 1 ||
      torch::utils::THP_nativeByteOrder() ==
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
    doRead(file, data, storage->nbytes());
  } else {
    int64_t buffer_size = std::min(size, (int64_t)5000);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    std::unique_ptr<uint8_t[]> le_buffer(
        new uint8_t[buffer_size * element_size]);

    for (int64_t i = 0; i < size; i += buffer_size) {
      size_t to_convert = std::min(size - i, buffer_size);
      doRead(file, le_buffer.get(), element_size * to_convert);

      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (element_size == 2) {
        torch::utils::THP_decodeInt16Buffer(
            (int16_t*)data + i, le_buffer.get(), true, to_convert);
      } else if (element_size == 4) {
        torch::utils::THP_decodeInt32Buffer(
            (int32_t*)data + i, le_buffer.get(), true, to_convert);
      } else if (element_size == 8) {
        torch::utils::THP_decodeInt64Buffer(
            (int64_t*)data + i, le_buffer.get(), true, to_convert);
      }
    }
  }

  if (storage->device_type() != at::kCPU) {
    // Here we use a tensor.copy_() to impl H2D for all non-CPU device.
    auto cpu_tensor = at::from_blob(
        (void*)data,
        {static_cast<int64_t>(nbytes)},
        at::device(at::kCPU).dtype(c10::kByte));
    auto device_tensor = at::from_blob(
        storage->mutable_data(),
        {static_cast<int64_t>(nbytes)},
        {1},
        nullptr,
        at::device(storage->device()).dtype(c10::kByte),
        {storage->device()});
    device_tensor.copy_(cpu_tensor);
  }
  return storage;
}

template c10::intrusive_ptr<c10::StorageImpl> THPStorage_readFileRaw<int>(
    int fd,
    c10::intrusive_ptr<c10::StorageImpl> storage,
    uint64_t element_size);
template c10::intrusive_ptr<c10::StorageImpl> THPStorage_readFileRaw<PyObject*>(
    PyObject* fd,
    c10::intrusive_ptr<c10::StorageImpl> storage,
    uint64_t element_size);
