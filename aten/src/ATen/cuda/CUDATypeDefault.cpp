#include <ATen/cuda/CUDATypeDefault.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADevice.h>

namespace at {

Allocator* CUDATypeDefault::allocator() const {
  return cuda::getCUDADeviceAllocator();
}
Device CUDATypeDefault::getDeviceFromPtr(void * data) const {
  return cuda::getDeviceFromPtr(data);
}

} // namespace at
