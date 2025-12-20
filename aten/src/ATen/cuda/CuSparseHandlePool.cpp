#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

namespace at::cuda {
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
} // namespace

cusparseHandle_t getCurrentCUDASparseHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  auto handle = at::cuda::reserveHandle<cusparseHandle_t, createCusparseHandle, destroyCusparseHandle>(device);
  TORCH_CUDASPARSE_CHECK(cusparseSetStream(handle, c10::cuda::getCurrentCUDAStream()));
  return handle;
}

} // namespace at::cuda
