#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

namespace at {
namespace native {

// this needs to be split along CPU/CUDA lines because we don't have a consistent
// way of getting the allocator to use for a device (c10::GetAllocator is not
// the same as at::cuda::getCUDADeviceAllocator().
Tensor& set_cuda_(Tensor& result) {
  Storage storage(result.dtype(), 0, at::cuda::getCUDADeviceAllocator(), true);
  return result.set_(storage, 0, {0}, {});
}

}
}
