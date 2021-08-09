#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <THC/THC.h>  // for USE_MAGMA

#ifdef USE_MAGMA
#include <magma_types.h>
#include <magma_v2.h>
#endif

namespace at {
namespace native {

#ifdef USE_MAGMA

// RAII for a MAGMA Queue
struct MAGMAQueue {

  // Default constructor without a device will cause
  // destroying a queue which has not been initialized.
  MAGMAQueue() = delete;

  // Constructor
  explicit MAGMAQueue(int64_t device_id) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
#if CUDA_VERSION >= 11000
    // Magma operations is numerically sensitive, so TF32 should be off
    // regardless of the global flag.
    TORCH_CUDABLAS_CHECK(cublasGetMathMode(handle, &original_math_mode));
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
#endif
    magma_queue_create_from_cuda(
      device_id,
      at::cuda::getCurrentCUDAStream(),
      handle,
      at::cuda::getCurrentCUDASparseHandle(),
      &magma_queue_);
  }

  // Getter
  magma_queue_t get_queue() const { return magma_queue_; }

  // Destructor
  ~MAGMAQueue() {
#if CUDA_VERSION >= 11000
    // We've manually set the math mode to CUBLAS_DEFAULT_MATH, now we
    // should restore the original math mode back
    cublasHandle_t handle = magma_queue_get_cublas_handle(magma_queue_);
    cublasSetMathMode(handle, original_math_mode);
#endif
    magma_queue_destroy(magma_queue_);
  }

 private:
  magma_queue_t magma_queue_;
#if CUDA_VERSION >= 11000
  cublasMath_t original_math_mode;
#endif
};

static inline magma_int_t magma_int_cast(int64_t value, const char* varname) {
  auto result = static_cast<magma_int_t>(value);
  if (static_cast<int64_t>(result) != value) {
    AT_ERROR("magma: The value of ", varname, "(", (long long)value,
             ") is too large to fit into a magma_int_t (", sizeof(magma_int_t), " bytes)");
  }
  return result;
}

// MAGMA functions that don't take a magma_queue_t aren't stream safe
// Work around this by synchronizing with the default stream
struct MagmaStreamSyncGuard {
  MagmaStreamSyncGuard() {
    auto stream = at::cuda::getCurrentCUDAStream();
    if (stream != at::cuda::getDefaultCUDAStream()) {
      at::cuda::stream_synchronize(stream);
    }
  }

  ~MagmaStreamSyncGuard() noexcept(false) {
    auto default_stream = at::cuda::getDefaultCUDAStream();
    if (at::cuda::getCurrentCUDAStream() != default_stream) {
      at::cuda::stream_synchronize(default_stream);
    }
  }
};
#endif

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

} // namespace native
} // namespace at
