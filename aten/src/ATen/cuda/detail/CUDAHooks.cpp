#include <ATen/cuda/detail/CUDAHooks.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/CUDAGenerator.h>
#include <ATen/RegisterCUDA.h>

#include "THC/THC.h"

#include <cuda.h>

namespace at { namespace cuda { namespace detail {

// NB: deleter is dynamic, because we need it to live in a separate
// compilation unit (alt is to have another method in hooks, but
// let's not if we don't need to!)
std::unique_ptr<THCState, void(*)(THCState*)> CUDAHooks::initCUDA() const {
  THCState* thc_state = THCState_alloc();
  THCState_setDeviceAllocator(thc_state, THCCachingAllocator_get());
  thc_state->cudaHostAllocator = &THCCachingHostAllocator;
  THCudaInit(thc_state);
  return std::unique_ptr<THCState, void(*)(THCState*)>(thc_state, [](THCState* p) {
        if (p) THCState_free(p);
      });
}

std::unique_ptr<Generator> CUDAHooks::initCUDAGenerator(Context* context) const {
  return std::unique_ptr<Generator>(new CUDAGenerator(context));
}

bool CUDAHooks::hasCUDA() const {
  int count;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err == cudaErrorInsufficientDriver) {
    return false;
  }
  return true;
}

cudaStream_t CUDAHooks::getCurrentCUDAStream(THCState* thc_state) const {
  return THCState_getCurrentStream(thc_state);
}
struct cudaDeviceProp* CUDAHooks::getCurrentDeviceProperties(THCState* thc_state) const {
  return THCState_getCurrentDeviceProperties(thc_state);
}
struct cudaDeviceProp* CUDAHooks::getDeviceProperties(THCState* thc_state, int device) const {
  return THCState_getDeviceProperties(thc_state, device);
}

int64_t CUDAHooks::current_device() const {
  int device;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    return device;
  }
  return -1;
}

std::unique_ptr<Allocator> CUDAHooks::newPinnedMemoryAllocator() const {
  return std::unique_ptr<Allocator>(new PinnedMemoryAllocator());
}

void CUDAHooks::registerCUDATypes(Context* context) const {
  register_cuda_types(context);
}

}}} // namespace at::cuda::detail
