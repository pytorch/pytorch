#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/serialization.cpp"
#else

#define SYSCHECK(call) { ssize_t __result = call; if (__result < 0) throw std::system_error(__result, std::system_category()); }

void THPTensor_(writeMetadataRaw)(THTensor *self, int fd)
{
  SYSCHECK(write(fd, &self->nDimension, sizeof(long)));
  SYSCHECK(write(fd, self->size, sizeof(long) * self->nDimension));
  SYSCHECK(write(fd, self->stride, sizeof(long) * self->nDimension));
  SYSCHECK(write(fd, &self->storageOffset, sizeof(long)));
}

THTensor * THPTensor_(newWithMetadataFileRaw)(int fd, THStorage *storage)
{
  THTensorPtr tensor = THTensor_(new)(LIBRARY_STATE_NOARGS);
  SYSCHECK(read(fd, &tensor->nDimension, sizeof(long)));
  tensor->size = (long*)THAlloc(tensor->nDimension * sizeof(long));
  tensor->stride = (long*)THAlloc(tensor->nDimension * sizeof(long));
  SYSCHECK(read(fd, tensor->size, sizeof(long) * tensor->nDimension));
  SYSCHECK(read(fd, tensor->stride, sizeof(long) * tensor->nDimension));
  SYSCHECK(read(fd, &tensor->storageOffset, sizeof(long)));
  THStorage_(retain)(LIBRARY_STATE storage);
  tensor->storage = storage;
  return tensor.release();
}

void THPStorage_(writeFileRaw)(THStorage *self, int fd)
{
  real *data;
#ifndef THC_GENERIC_FILE
  data = self->data;
#else
  std::unique_ptr<char[]> cpu_data(new char[self->size * sizeof(real)]);
  data = (real*)cpu_data.get();
  THCudaCheck(cudaMemcpy(data, self->data, self->size * sizeof(real), cudaMemcpyDeviceToHost));
#endif
  SYSCHECK(write(fd, &self->size, sizeof(long)));
  // fast track for bytes and little endian
  if (sizeof(real) == 1 || THP_nativeByteOrder() == THPByteOrder::THP_LITTLE_ENDIAN) {
    char *bytes = (char *) data;
    uint64_t remaining = sizeof(real) * self->size;
    while (remaining > 0) {
      ssize_t result = write(fd, bytes, remaining);
      if (result < 0)
        throw std::system_error(result, std::system_category());
      bytes += result;
      remaining -= result;
    }
  } else {
    int64_t buffer_size = std::min(self->size, (long)5000);
    std::unique_ptr<uint8_t[]> le_buffer(new uint8_t[buffer_size * sizeof(real)]);
    for (int64_t i = 0; i < self->size; i += buffer_size) {
      size_t to_convert = std::min(self->size - i, buffer_size);
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
      SYSCHECK(write(fd, le_buffer.get(), to_convert * sizeof(real)));
    }
  }
}

THStorage * THPStorage_(readFileRaw)(int fd)
{
  real *data;
  long size;
  SYSCHECK(read(fd, &size, sizeof(long)));
  THStoragePtr storage = THStorage_(newWithSize)(LIBRARY_STATE size);

#ifndef THC_GENERIC_FILE
  data = storage->data;
#else
  std::unique_ptr<char[]> cpu_data(new char[size * sizeof(real)]);
  data = (real*)cpu_data.get();
#endif

  // fast track for bytes and little endian
  if (sizeof(real) == 1 || THP_nativeByteOrder() == THPByteOrder::THP_LITTLE_ENDIAN) {
    char *bytes = (char *) data;
    uint64_t remaining = sizeof(real) * storage->size;
    while (remaining > 0) {
      ssize_t result = read(fd, bytes, remaining);
      if (result < 0)
        throw std::system_error(result, std::system_category());
      bytes += result;
      remaining -= result;
    }
  } else {
    int64_t buffer_size = std::min(size, (long)5000);
    std::unique_ptr<uint8_t[]> le_buffer(new uint8_t[buffer_size * sizeof(real)]);
    for (int64_t i = 0; i < size; i += buffer_size) {
      size_t to_convert = std::min(size - i, buffer_size);
      SYSCHECK(read(fd, le_buffer.get(), sizeof(real) * to_convert));
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
  THCudaCheck(cudaMemcpy(storage->data, data, size * sizeof(real), cudaMemcpyHostToDevice));
#endif
  return storage.release();
}

#undef SYSCHECK

#endif
