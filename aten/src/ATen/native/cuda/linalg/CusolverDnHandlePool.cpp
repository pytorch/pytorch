#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

namespace at::cuda {
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
  (void)handle; // Suppress unused variable warning
#else
    cusolverDnDestroy(handle);
#endif
}
} // namespace

cusolverDnHandle_t getCurrentCUDASolverDnHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  auto handle = at::cuda::reserveHandle<cusolverDnHandle_t, createCusolverDnHandle, destroyCusolverDnHandle>(device);
  auto stream = c10::cuda::getCurrentCUDAStream();
  TORCH_CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return handle;
}

} // namespace at::cuda
