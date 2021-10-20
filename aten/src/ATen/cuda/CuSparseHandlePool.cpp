#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

namespace at { namespace cuda {
namespace {

void createCusparseHandle(cusparseHandle_t *handle) {
  TORCH_CUDASPARSE_CHECK(cusparseCreate(handle));
}

void destroyCusparseHandle(cusparseHandle_t handle) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the cuda context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and @soumith decided to not destroy
// the handle as a workaround.
//   - Comments of @soumith copied from cuDNN handle pool implementation
#ifdef NO_CUDNN_DESTROY_HANDLE
#else
    cusparseDestroy(handle);
#endif
}

using CuSparsePoolType = DeviceThreadHandlePool<cusparseHandle_t, createCusparseHandle, destroyCusparseHandle>;

} // namespace

cusparseHandle_t getCurrentCUDASparseHandle() {
  int device;
  AT_CUDA_CHECK(cudaGetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<CuSparsePoolType>();
  thread_local std::unique_ptr<CuSparsePoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  TORCH_CUDASPARSE_CHECK(cusparseSetStream(handle, c10::cuda::getCurrentCUDAStream()));
  return handle;
}

}} // namespace at::cuda
