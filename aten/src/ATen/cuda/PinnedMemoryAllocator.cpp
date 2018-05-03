#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/Context.h>
#include <ATen/Config.h>

#include <THC/THC.h>

#include <stdexcept>

namespace at { namespace cuda {

void* PinnedMemoryAllocator::allocate(std::size_t n) const {
  auto state = globalContext().lazyInitCUDA();
  return state->cudaHostAllocator->malloc(nullptr, n);
}

void PinnedMemoryAllocator::deallocate(void* ptr) const {
  auto state = globalContext().lazyInitCUDA();
  return state->cudaHostAllocator->free(nullptr, ptr);
}

}} // namespace at::cuda
