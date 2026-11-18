#pragma once
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>


namespace at::native {

static inline int cuda_int_cast(int64_t value, const char* varname) {
  auto result = static_cast<int>(value);
  TORCH_CHECK(static_cast<int64_t>(result) == value,
              "cuda_int_cast: The value of ", varname, "(", (long long)value,
              ") is too large to fit into a int (", sizeof(int), " bytes)");
  return result;
}

// Creates an array of size elements of type T, backed by pinned memory
// wrapped in a Storage
template<class T>
static inline Storage pin_memory(int64_t size) {
  auto* allocator = cuda::getPinnedMemoryAllocator();
  int64_t adjusted_size = size * sizeof(T);
  return Storage(
      Storage::use_byte_size_t(),
      adjusted_size,
      allocator,
      /*resizable=*/false);
}

} // namespace at::native
