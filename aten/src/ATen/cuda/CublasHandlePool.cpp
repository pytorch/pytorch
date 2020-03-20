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

DeviceThreadHandlePool<cublasHandle_t, createCublasHandle, destroyCublasHandle> pool;

// Thread local PoolWindows are wrapped by unique_ptrs and lazily-initialized
// to avoid initialization issues that caused hangs on Windows.
// See: https://github.com/pytorch/pytorch/pull/22405
// This thread local unique_ptrs will be destroyed when the thread terminates,
// releasing its reserved handles back to the pool.
thread_local std::unique_ptr<decltype(pool)::PoolWindow> myPoolWindow;

} // namespace

cublasHandle_t getCurrentCUDABlasHandle() {
  int device;
  AT_CUDA_CHECK(cudaGetDevice(&device));

  if (!myPoolWindow)
    myPoolWindow.reset(pool.newPoolWindow());
  auto handle = myPoolWindow->reserve(device);
  auto stream = c10::cuda::getCurrentCUDAStream();
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
  return handle;
}

}} // namespace at::cuda
