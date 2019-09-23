#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "torch/csrc/generic/serialization.cpp"
#else

#ifdef THC_GENERIC_FILE
#include <c10/cuda/CUDAGuard.h>
#endif

template <class io>
void THPStorage_(writeFileRaw)(THWStorage *self, io fd)
{
#ifdef THC_GENERIC_FILE
  c10::cuda::CUDAGuard guard(self->device());
#endif

  scalar_t *data;
  int64_t size = THWStorage_(size)(LIBRARY_STATE self);
#ifndef THC_GENERIC_FILE
  data = THWStorage_(data)(LIBRARY_STATE self);
#else
  std::unique_ptr<char[]> cpu_data(new char[size * sizeof(scalar_t)]);
  data = (scalar_t*)cpu_data.get();
  THCudaCheck(cudaMemcpy(data, THWStorage_(data)(LIBRARY_STATE self), size * sizeof(scalar_t), cudaMemcpyDeviceToHost));
#endif
  if (torch::utils::THP_nativeByteOrder() ==
      torch::utils::THPByteOrder::THP_LITTLE_ENDIAN)
    doWrite(fd, &size, sizeof(int64_t));
  else {
    int64_t nsize; // convert big endian cpu to little endian storage
    torch::utils::THP_encodeInt64Buffer(
        (uint8_t*)&nsize,
        (const int64_t*)&size,
        torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
        1);
    doWrite(fd, &nsize, sizeof(int64_t));
  }
  // fast track for bytes and little endian
  if (sizeof(scalar_t) == 1 ||
      torch::utils::THP_nativeByteOrder() ==
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
    doWrite(fd, data, sizeof(scalar_t) * size);
  } else {
    int64_t buffer_size = std::min(size, (int64_t)5000);
    std::unique_ptr<uint8_t[]> le_buffer(new uint8_t[buffer_size * sizeof(scalar_t)]);
    for (int64_t i = 0; i < size; i += buffer_size) {
      size_t to_convert = std::min(size - i, buffer_size);
      if (sizeof(scalar_t) == 2) {
        torch::utils::THP_encodeInt16Buffer(
            (uint8_t*)le_buffer.get(),
            (const int16_t*)data + i,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (sizeof(scalar_t) == 4) {
        torch::utils::THP_encodeInt32Buffer(
            (uint8_t*)le_buffer.get(),
            (const int32_t*)data + i,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      } else if (sizeof(scalar_t) == 8) {
        torch::utils::THP_encodeInt64Buffer(
            (uint8_t*)le_buffer.get(),
            (const int64_t*)data + i,
            torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
            to_convert);
      }
      doWrite(fd, le_buffer.get(), to_convert * sizeof(scalar_t));
    }
  }
}

template void THPStorage_(writeFileRaw<int>)(THWStorage *self, int fd);
template void THPStorage_(writeFileRaw<PyObject*>)(THWStorage *self, PyObject* fd);

template <class io>
THWStorage * THPStorage_(readFileRaw)(io file, THWStorage *_storage)
{
#ifdef THC_GENERIC_FILE
  c10::cuda::OptionalCUDAGuard guard;
  if (_storage != nullptr) {
    guard.set_device(_storage->device());
  }
#endif

  scalar_t *data;
  int64_t size;
  doRead(file, &size, sizeof(int64_t));
  if (torch::utils::THP_nativeByteOrder() ==
      torch::utils::THPByteOrder::THP_BIG_ENDIAN) {
    int64_t nsize; // convert little endian storage to big endian cpu
    nsize = size;
    torch::utils::THP_decodeInt64Buffer(
        &size, (const uint8_t*)&nsize, torch::utils::THP_nativeByteOrder(), 1);
  }
  THWStoragePtr storage;
  if (_storage == nullptr) {
    storage = THWStorage_(newWithSize)(LIBRARY_STATE size);
  } else {
    THPUtils_assert(THWStorage_(size)(LIBRARY_STATE _storage) == size,
        "storage has wrong size: expected %ld got %ld",
        size, THWStorage_(size)(LIBRARY_STATE _storage));
    storage = _storage;
  }

#ifndef THC_GENERIC_FILE
  data = THWStorage_(data)(LIBRARY_STATE storage);
#else
  std::unique_ptr<char[]> cpu_data(new char[size * sizeof(scalar_t)]);
  data = (scalar_t*)cpu_data.get();
#endif

  // fast track for bytes and little endian
  if (sizeof(scalar_t) == 1 ||
      torch::utils::THP_nativeByteOrder() ==
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
    doRead(file, data, sizeof(scalar_t) * THWStorage_(size)(LIBRARY_STATE storage));
  } else {
    int64_t buffer_size = std::min(size, (int64_t)5000);
    std::unique_ptr<uint8_t[]> le_buffer(new uint8_t[buffer_size * sizeof(scalar_t)]);


    for (int64_t i = 0; i < size; i += buffer_size) {
      size_t to_convert = std::min(size - i, buffer_size);
      doRead(file, le_buffer.get(), sizeof(scalar_t) * to_convert);

      if (sizeof(scalar_t) == 2) {
        torch::utils::THP_decodeInt16Buffer(
            (int16_t*)data + i,
            le_buffer.get(),
            torch::utils::THP_nativeByteOrder(),
            to_convert);
      } else if (sizeof(scalar_t) == 4) {
        torch::utils::THP_decodeInt32Buffer(
            (int32_t*)data + i,
            le_buffer.get(),
            torch::utils::THP_nativeByteOrder(),
            to_convert);
      } else if (sizeof(scalar_t) == 8) {
        torch::utils::THP_decodeInt64Buffer(
            (int64_t*)data + i,
            le_buffer.get(),
            torch::utils::THP_nativeByteOrder(),
            to_convert);
      }
    }
  }

#ifdef THC_GENERIC_FILE
  THCudaCheck(cudaMemcpy(THWStorage_(data)(LIBRARY_STATE storage), data, size * sizeof(scalar_t), cudaMemcpyHostToDevice));
#endif
  return storage.release();
}

template THWStorage* THPStorage_(readFileRaw<int>)(int fd, THWStorage* storage);
template THWStorage* THPStorage_(readFileRaw<PyObject*>)(PyObject* fd, THWStorage* storage);

#endif
