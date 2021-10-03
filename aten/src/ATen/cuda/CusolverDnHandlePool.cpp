#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

#ifdef CUDART_VERSION

namespace at { namespace cuda {
namespace {

void createCusolverDnHandle(cusolverDnHandle_t *handle) {
  TORCH_CUSOLVER_CHECK(cusolverDnCreate(handle));
}

void destroyCusolverDnHandle(cusolverDnHandle_t handle) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the cuda context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and @soumith decided to not destroy
// the handle as a workaround.
//   - Comments of @soumith copied from cuDNN handle pool implementation
#ifdef NO_CUDNN_DESTROY_HANDLE
#else
    cusolverDnDestroy(handle);
#endif
}

using CuSolverDnPoolType = DeviceThreadHandlePool<cusolverDnHandle_t, createCusolverDnHandle, destroyCusolverDnHandle>;

} // namespace

cusolverDnHandle_t getCurrentCUDASolverDnHandle() {
  int device;
  AT_CUDA_CHECK(cudaGetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<CuSolverDnPoolType>();
  thread_local std::unique_ptr<CuSolverDnPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  auto stream = c10::cuda::getCurrentCUDAStream();
  TORCH_CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return handle;
}

}} // namespace at::cuda

#endif // CUDART_VERSION
