#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "torch/csrc/generic/serialization.cpp"
#else

#ifdef THC_GENERIC_FILE
#include <c10/cuda/CUDAGuard.h>
#endif

// save_save is necessary since the old eager format saved storages as
// [size + data], but the v1.5 eager format removes this since size is saved in
// the filesize.
template <class io>
void THPStorage_(writeFileRaw)(THWStorage *self, io fd, bool save_size)
{
#ifdef THC_GENERIC_FILE
  c10::cuda::CUDAGuard guard(self->device());
#endif

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  scalar_t *data;
  int64_t size_bytes = self->nbytes();
  int64_t numel = size_bytes / sizeof(scalar_t);
#ifndef THC_GENERIC_FILE
  data = THWStorage_(data)(LIBRARY_STATE self);
#else
  std::unique_ptr<char[]> cpu_data(new char[size_bytes]);
  data = (scalar_t*)cpu_data.get();
  THCudaCheck(cudaMemcpy(
      data,
      THWStorage_(data)(LIBRARY_STATE self),
      size_bytes,
      cudaMemcpyDeviceToHost));
#endif
  if (save_size) {
    if (torch::utils::THP_nativeByteOrder() ==
        torch::utils::THPByteOrder::THP_LITTLE_ENDIAN)
      doWrite(fd, &numel, sizeof(int64_t));
    else {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t nsize; // convert big endian cpu to little endian storage
      torch::utils::THP_encodeInt64Buffer(
          (uint8_t*)&nsize,
          (const int64_t*)&numel,
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN,
          1);
      doWrite(fd, &nsize, sizeof(int64_t));
    }
  }
  // fast track for bytes and little endian
  if (sizeof(scalar_t) == 1 ||
      torch::utils::THP_nativeByteOrder() ==
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
    doWrite(fd, data, size_bytes);
  } else {
    int64_t buffer_size = std::min(numel, (int64_t)5000);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    std::unique_ptr<uint8_t[]> le_buffer(new uint8_t[buffer_size * sizeof(scalar_t)]);
    for (int64_t i = 0; i < numel; i += buffer_size) {
      size_t to_convert = std::min(numel - i, buffer_size);
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

template void THPStorage_(writeFileRaw<int>)(THWStorage *self, int fd, bool save_size);
template void THPStorage_(writeFileRaw<PyObject*>)(THWStorage *self, PyObject* fd, bool save_size);

template <class io>
THWStorage * THPStorage_(readFileRaw)(io file, THWStorage *_storage)
{
#ifdef THC_GENERIC_FILE
  c10::cuda::OptionalCUDAGuard guard;
  if (_storage != nullptr) {
    guard.set_device(_storage->device());
  }
#endif

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  scalar_t *data;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t size;
  doRead(file, &size, sizeof(int64_t));
  if (torch::utils::THP_nativeByteOrder() ==
      torch::utils::THPByteOrder::THP_BIG_ENDIAN) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t nsize; // convert little endian storage to big endian cpu
    nsize = size;
    torch::utils::THP_decodeInt64Buffer(
        &size, (const uint8_t*)&nsize, torch::utils::THP_nativeByteOrder(), 1);
  }
  THWStoragePtr storage;
  if (_storage == nullptr) {
    storage = THWStorage_(newWithSize)(LIBRARY_STATE size);
  } else {
    int64_t _storage_numel = _storage->nbytes() / sizeof(scalar_t);
    THPUtils_assert(
        _storage_numel == size,
        "storage has wrong size: expected %ld got %ld",
        size,
        _storage_numel);
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
    doRead(file, data, storage->nbytes());
  } else {
    int64_t buffer_size = std::min(size, (int64_t)5000);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
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
