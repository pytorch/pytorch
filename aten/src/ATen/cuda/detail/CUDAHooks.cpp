#include <ATen/cuda/detail/CUDAHooks.h>

#include <cuda.h>
#include "THC/THC.h"
#include "ATen/CUDAGenerator.h"

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

}}} // namespace at::cuda::detail
