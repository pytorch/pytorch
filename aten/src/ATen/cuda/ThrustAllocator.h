#pragma once

#include <cstddef>
#include <c10/cuda/CUDACachingAllocator.h>

namespace at::cuda {

/// Allocator for Thrust to re-route its internal device allocations
/// to the THC allocator
class ThrustAllocator {
public:
  typedef char value_type;

  char* allocate(std::ptrdiff_t size) {
    return static_cast<char*>(c10::cuda::CUDACachingAllocator::raw_allocate(size));
  }

  void deallocate(char* p, size_t size) {
    c10::cuda::CUDACachingAllocator::raw_deallocate(p);
  }
};

} // namespace at::cuda
