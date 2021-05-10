#include <ATen/Config.h>
#include <ATen/Context.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>

#include <THC/THC.h>
#include <THC/THCGeneral.hpp>

#include <stdexcept>

namespace at {
namespace cuda {

at::Allocator* getPinnedMemoryAllocator() {
  auto state = globalContext().lazyInitCUDA();
  return state->cudaHostAllocator;
}

} // namespace cuda
} // namespace at
