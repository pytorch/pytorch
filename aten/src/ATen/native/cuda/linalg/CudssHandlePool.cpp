#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

#if defined(USE_CUDSS)

namespace at::cuda {
namespace {

void createCudssHandle(cudssHandle_t *handle) {
  TORCH_CUDSS_CHECK(cudssCreate(handle));
}

void destroyCudssHandle(cudssHandle_t handle) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the cuda context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and @soumith decided to not destroy
// the handle as a workaround.
//   - Comments of @soumith copied from cuDNN handle pool implementation
#ifdef NO_CUDNN_DESTROY_HANDLE
  (void)handle; // Suppress unused variable warning
#else
    cudssDestroy(handle);
#endif
}

using CudssPoolType = DeviceThreadHandlePool<cudssHandle_t, createCudssHandle, destroyCudssHandle>;

} // namespace

cudssHandle_t getCurrentCudssHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<CudssPoolType>();
  thread_local std::unique_ptr<CudssPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  auto stream = c10::cuda::getCurrentCUDAStream();
  TORCH_CUDSS_CHECK(cudssSetStream(handle, stream));
  return handle;
}

} // namespace at::cuda

#endif
