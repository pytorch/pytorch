#include <ATen/cuda/detail/DeviceThreadHandles.h>
#include <ATen/cudnn/Handle.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/cuda/Exceptions.h>

namespace at::native {
namespace {

void createCuDNNHandle(cudnnHandle_t* handle) {
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
} // namespace

cudnnHandle_t getCudnnHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  auto handle = at::cuda::reserveHandle<cudnnHandle_t, createCuDNNHandle, destroyCuDNNHandle>(device);
  AT_CUDNN_CHECK(cudnnSetStream(handle, c10::cuda::getCurrentCUDAStream()));
  return handle;
}

} // namespace at::native
