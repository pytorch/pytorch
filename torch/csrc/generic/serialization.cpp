#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/serialization.cpp"
#else

#define SYSCHECK(call) { ssize_t __result = call; if (__result < 0) throw std::system_error((int) __result, std::system_category()); }

template <class io>
void THPStorage_(writeFileRaw)(THStorage *self, io fd)
{
  real *data;
  int64_t size = THStorage_(size)(LIBRARY_STATE self);
#ifndef THC_GENERIC_FILE
  data = THStorage_(data)(LIBRARY_STATE self);
#else
  std::unique_ptr<char[]> cpu_data(new char[size * sizeof(real)]);
  data = (real*)cpu_data.get();
  THCudaCheck(cudaMemcpy(data, THStorage_(data)(LIBRARY_STATE self), size * sizeof(real), cudaMemcpyDeviceToHost));
#endif
  ssize_t result = doWrite(fd, &size, sizeof(int64_t));
  if (result != sizeof(int64_t))
    throw std::system_error(result, std::system_category());
  // fast track for bytes and little endian
  if (sizeof(real) == 1 || THP_nativeByteOrder() == THPByteOrder::THP_LITTLE_ENDIAN) {
    char *bytes = (char *) data;
    int64_t remaining = sizeof(real) * size;
    while (remaining > 0) {
      // we write and read in 1GB blocks to avoid bugs on some OSes
      ssize_t result = doWrite(fd, bytes, THMin(remaining, 1073741824));
      if (result < 0)
        throw std::system_error(result, std::system_category());
      bytes += result;
      remaining -= result;
    }
    if (remaining != 0)
      throw std::system_error(result, std::system_category());
  } else {
    int64_t buffer_size = std::min(size, (int64_t)5000);
    std::unique_ptr<uint8_t[]> le_buffer(new uint8_t[buffer_size * sizeof(real)]);
    for (int64_t i = 0; i < size; i += buffer_size) {
      size_t to_convert = std::min(size - i, buffer_size);
      if (sizeof(real) == 2) {
        THP_encodeInt16Buffer((uint8_t*)le_buffer.get(),
            (const int16_t*)data + i,
            THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (sizeof(real) == 4) {
        THP_encodeInt32Buffer((uint8_t*)le_buffer.get(),
            (const int32_t*)data + i,
            THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (sizeof(real) == 8) {
        THP_encodeInt64Buffer((uint8_t*)le_buffer.get(),
            (const int64_t*)data + i,
            THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      }
      SYSCHECK(doWrite(fd, le_buffer.get(), to_convert * sizeof(real)));
    }
  }
}

template void THPStorage_(writeFileRaw<int>)(THStorage *self, int fd);
template void THPStorage_(writeFileRaw<PyObject*>)(THStorage *self, PyObject* fd);

template <class io>
THStorage * THPStorage_(readFileRaw)(io file, THStorage *_storage)
{
  real *data;
  int64_t size;
  ssize_t result = doRead(file, &size, sizeof(int64_t));
  if (result == 0)
    throw std::runtime_error("unexpected EOF. The file might be corrupted.");
  if (result != sizeof(int64_t))
    throw std::system_error(result, std::system_category());
  THStoragePtr storage;
  if (_storage == nullptr) {
    storage = THStorage_(newWithSize)(LIBRARY_STATE size);
  } else {
    THPUtils_assert(THStorage_(size)(LIBRARY_STATE _storage) == size,
        "storage has wrong size: expected %ld got %ld",
        size, THStorage_(size)(LIBRARY_STATE _storage));
    storage = _storage;
  }

#ifndef THC_GENERIC_FILE
  data = THStorage_(data)(LIBRARY_STATE storage);
#else
  std::unique_ptr<char[]> cpu_data(new char[size * sizeof(real)]);
  data = (real*)cpu_data.get();
#endif

  // fast track for bytes and little endian
  if (sizeof(real) == 1 || THP_nativeByteOrder() == THPByteOrder::THP_LITTLE_ENDIAN) {
    char *bytes = (char *) data;
    int64_t remaining = sizeof(real) * THStorage_(size)(LIBRARY_STATE storage);
    while (remaining > 0) {
      // we write and read in 1GB blocks to avoid bugs on some OSes
      ssize_t result = doRead(file, bytes, THMin(remaining, 1073741824));
      if (result == 0) // 0 means EOF, which is also an error
        throw std::runtime_error("unexpected EOF. The file might be corrupted.");
      if (result < 0)
        throw std::system_error(result, std::system_category());
      bytes += result;
      remaining -= result;
    }
    if (remaining != 0)
      throw std::system_error(result, std::system_category());
  } else {
    int64_t buffer_size = std::min(size, (int64_t)5000);
    std::unique_ptr<uint8_t[]> le_buffer(new uint8_t[buffer_size * sizeof(real)]);


    for (int64_t i = 0; i < size; i += buffer_size) {
      size_t to_convert = std::min(size - i, buffer_size);
      SYSCHECK(doRead(file, le_buffer.get(), sizeof(real) * to_convert));

      if (sizeof(real) == 2) {
        THP_decodeInt16Buffer((int16_t*)data + i,
            le_buffer.get(),
            THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (sizeof(real) == 4) {
        THP_decodeInt32Buffer((int32_t*)data + i,
            le_buffer.get(),
            THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (sizeof(real) == 8) {
        THP_decodeInt64Buffer((int64_t*)data + i,
            le_buffer.get(),
            THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      }
    }
  }

#ifdef THC_GENERIC_FILE
  THCudaCheck(cudaMemcpy(THStorage_(data)(LIBRARY_STATE storage), data, size * sizeof(real), cudaMemcpyHostToDevice));
#endif
  return storage.release();
}

template THStorage* THPStorage_(readFileRaw<int>)(int fd, THStorage* storage);
template THStorage* THPStorage_(readFileRaw<PyObject*>)(PyObject* fd, THStorage* storage);

#undef SYSCHECK

#endif
