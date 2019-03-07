#include <ATen/cuda/CUDATypeDefault.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/CUDAGenerator.h>

namespace at {

Allocator* CUDATypeDefault::allocator() const {
  return cuda::getCUDADeviceAllocator();
}
Device CUDATypeDefault::getDeviceFromPtr(void * data) const {
  return cuda::getDeviceFromPtr(data);
}
std::unique_ptr<Generator> CUDATypeDefault::generator() const {
  return std::unique_ptr<Generator>(new CUDAGenerator(&at::globalContext()));
}

} // namespace at
