#include <ATen/cudnn/Handle.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>
#include <c10/cuda/CUDAStream.h>

namespace at { namespace native {
namespace {

void createCuDNNHandle(cudnnHandle_t *handle) {
  AT_CUDNN_CHECK(cudnnCreate(handle));
}

void destroyCuDNNHandle(cudnnHandle_t /*handle*/) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the cuda context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and I decided to not destroy
// the handle as a workaround.
//   - @soumith
//
// Further note: this is now disabled globally, because we are seeing
// the same issue as mentioned above in CUDA 11 CI.
//   - @zasdfgbnm
//
// #ifdef NO_CUDNN_DESTROY_HANDLE
// #else
//   cudnnDestroy(handle);
// #endif
}

using CudnnPoolType = at::cuda::DeviceThreadHandlePool<cudnnHandle_t, createCuDNNHandle, destroyCuDNNHandle>;

} // namespace

cudnnHandle_t getCudnnHandle() {
  int device;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<CudnnPoolType>();
  thread_local std::unique_ptr<CudnnPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  AT_CUDNN_CHECK(cudnnSetStream(handle, c10::cuda::getCurrentCUDAStream()));
  return handle;
}

}} // namespace at::native
