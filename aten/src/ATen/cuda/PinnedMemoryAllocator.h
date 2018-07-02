#pragma once

#include <ATen/Allocator.h>

namespace at { namespace cuda {

struct PinnedMemoryAllocator final : public Allocator {
  void* allocate(size_t n) const override;
  void deallocate(void* ptr) const override;
};

PinnedMemoryAllocator* getPinnedMemoryAllocator();

}} // namespace at::cuda
