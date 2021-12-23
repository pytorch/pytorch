#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "torch/csrc/generic/serialization.cpp"
#else

#ifdef THC_GENERIC_FILE
#include <c10/cuda/CUDAGuard.h>
#else
#include <c10/core/CPUAllocator.h>
#endif

// save_save is necessary since the old eager format saved storages as
// [size + data], but the v1.5 eager format removes this since size is saved in
// the filesize.
template <class io>
void THPStorage_(writeFileRaw)(c10::StorageImpl *self, io fd, bool save_size, uint64_t element_size)
{
#ifdef THC_GENERIC_FILE
  c10::cuda::CUDAGuard guard(self->device());
#endif

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  scalar_t *data;
  int64_t size_bytes = self->nbytes();
  int64_t numel = size_bytes / element_size;
#ifndef THC_GENERIC_FILE
  data = self->data<scalar_t>();
#else
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::unique_ptr<char[]> cpu_data(new char[size_bytes]);
  data = (scalar_t*)cpu_data.get();
  C10_CUDA_CHECK(cudaMemcpy(
      data,
      self->data<scalar_t>(),
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
  if (element_size == 1 ||
      torch::utils::THP_nativeByteOrder() ==
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
    doWrite(fd, data, size_bytes);
  } else {
    int64_t buffer_size = std::min(numel, (int64_t)5000);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::unique_ptr<uint8_t[]> le_buffer(new uint8_t[buffer_size * element_size]);
    for (int64_t i = 0; i < numel; i += buffer_size) {
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

template void THPStorage_(writeFileRaw<int>)(c10::StorageImpl *self, int fd, bool save_size, uint64_t element_size);
template void THPStorage_(writeFileRaw<PyObject*>)(c10::StorageImpl *self, PyObject* fd, bool save_size, uint64_t element_size);

template <class io>
c10::intrusive_ptr<c10::StorageImpl> THPStorage_(readFileRaw)(
    io file, c10::intrusive_ptr<c10::StorageImpl> storage, uint64_t element_size)
{
#ifdef THC_GENERIC_FILE
  c10::cuda::OptionalCUDAGuard guard;
  if (storage.defined()) {
    guard.set_device(storage->device());
  }
#endif

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  scalar_t *data;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t size;
  doRead(file, &size, sizeof(int64_t));
  int64_t nbytes = element_size * size;
  if (torch::utils::THP_nativeByteOrder() ==
      torch::utils::THPByteOrder::THP_BIG_ENDIAN) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t nsize; // convert little endian storage to big endian cpu
    nsize = nbytes;
    torch::utils::THP_decodeInt64Buffer(
        &nbytes, (const uint8_t*)&nsize, torch::utils::THP_nativeByteOrder(), 1);
  }
  if (!storage.defined()) {
    storage = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      nbytes,
#if defined(THC_GENERIC_FILE)
      c10::cuda::CUDACachingAllocator::get(),
#else
      c10::GetDefaultCPUAllocator(),
#endif
      /*resizable=*/true);
  } else {
    int64_t _storage_nbytes = storage->nbytes();
    TORCH_CHECK(
        _storage_nbytes == nbytes,
        "storage has wrong byte size: expected %ld got %ld",
        nbytes,
        _storage_nbytes);
  }

#ifndef THC_GENERIC_FILE
  data = storage->data<scalar_t>();
#else
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::unique_ptr<char[]> cpu_data(new char[nbytes]);
  data = (scalar_t*)cpu_data.get();
#endif

  // fast track for bytes and little endian
  if (element_size == 1 ||
      torch::utils::THP_nativeByteOrder() ==
          torch::utils::THPByteOrder::THP_LITTLE_ENDIAN) {
    doRead(file, data, storage->nbytes());
  } else {
    int64_t buffer_size = std::min(size, (int64_t)5000);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::unique_ptr<uint8_t[]> le_buffer(new uint8_t[buffer_size * element_size]);

    for (int64_t i = 0; i < size; i += buffer_size) {
      size_t to_convert = std::min(size - i, buffer_size);
      doRead(file, le_buffer.get(), element_size * to_convert);

      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (element_size == 2) {
        torch::utils::THP_decodeInt16Buffer(
            (int16_t*)data + i,
            le_buffer.get(),
            torch::utils::THP_nativeByteOrder(),
            to_convert);
      } else if (element_size == 4) {
        torch::utils::THP_decodeInt32Buffer(
            (int32_t*)data + i,
            le_buffer.get(),
            torch::utils::THP_nativeByteOrder(),
            to_convert);
      } else if (element_size == 8) {
        torch::utils::THP_decodeInt64Buffer(
            (int64_t*)data + i,
            le_buffer.get(),
            torch::utils::THP_nativeByteOrder(),
            to_convert);
      }
    }
  }

#ifdef THC_GENERIC_FILE
  C10_CUDA_CHECK(cudaMemcpy(storage->data<scalar_t>(), data, nbytes, cudaMemcpyHostToDevice));
#endif
  return storage;
}

template c10::intrusive_ptr<c10::StorageImpl> THPStorage_(readFileRaw<int>)(
    int fd, c10::intrusive_ptr<c10::StorageImpl> storage, uint64_t element_size);
template c10::intrusive_ptr<c10::StorageImpl> THPStorage_(readFileRaw<PyObject*>)(
    PyObject* fd, c10::intrusive_ptr<c10::StorageImpl> storage, uint64_t element_size);

#endif
