#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

namespace at { namespace cuda {
namespace {

void createCublasHandle(cublasHandle_t *handle) {
  TORCH_CUDABLAS_CHECK(cublasCreate(handle));
}

void destroyCublasHandle(cublasHandle_t handle) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the cuda context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and @soumith decided to not destroy
// the handle as a workaround.
//   - Comments of @soumith copied from cuDNN handle pool implementation
#ifdef NO_CUDNN_DESTROY_HANDLE
#else
    cublasDestroy(handle);
#endif
}

using CuBlasPoolType = DeviceThreadHandlePool<cublasHandle_t, createCublasHandle, destroyCublasHandle>;

} // namespace

cublasHandle_t getCurrentCUDABlasHandle() {
  int device;
  AT_CUDA_CHECK(cudaGetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.

  // Use a leaky singleton for the pool following standard practice around
  // singletons: https://isocpp.org/wiki/faq/ctors#construct-on-first-use-v2
  static auto pool = std::shared_ptr<CuBlasPoolType>(
      new CuBlasPoolType(), [](CuBlasPoolType* p) {
        // Leak the memory.
      });
  thread_local std::unique_ptr<CuBlasPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  auto stream = c10::cuda::getCurrentCUDAStream();
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
#if CUDA_VERSION >= 11000
  // On CUDA >= 11, and architecture >= Ampere, cuBLAS can use TF32 to speedup
  // FP32 data type calculations based on the value of the allow_tf32 flag.
  // To enable TF32, set the math mode of the handle to CUBLAS_TF32_TENSOR_OP_MATH.
  if (!NoTF32Guard::should_disable_tf32() && at::globalContext().allowTF32CuBLAS()) {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  } else {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  }
#endif
#if defined(__HIP_PLATFORM_HCC__) && HIP_VERSION >= 308
  rocblas_atomics_mode rocblas_mode;
  if (at::globalContext().deterministicAlgorithms()) {
    rocblas_mode = rocblas_atomics_not_allowed;
  } else {
    rocblas_mode = rocblas_atomics_allowed;
  }
  TORCH_CUDABLAS_CHECK(rocblas_set_atomics_mode(handle, rocblas_mode));
#endif
  return handle;
}

}} // namespace at::cuda
