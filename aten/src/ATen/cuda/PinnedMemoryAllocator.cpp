#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/Context.h>
#include <ATen/Config.h>

#include <THC/THC.h>
#include <THC/THCGeneral.hpp>

#include <stdexcept>

namespace at { namespace cuda {

void* PinnedMemoryAllocator::allocate(size_t n) const {
  auto state = globalContext().lazyInitCUDA();
  return state->cudaHostAllocator->malloc(nullptr, n);
}

void PinnedMemoryAllocator::deallocate(void* ptr) const {
  auto state = globalContext().lazyInitCUDA();
  return state->cudaHostAllocator->free(nullptr, ptr);
}

// No risk of static initialization order fiasco
static PinnedMemoryAllocator r;
PinnedMemoryAllocator* getPinnedMemoryAllocator() {
  return &r;
}

}} // namespace at::cuda
