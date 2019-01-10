#pragma once

#include <ATen/Allocator.h>

namespace at { namespace cuda {

struct PinnedMemoryAllocator final : public Allocator {
  void* allocate(std::size_t n) const override;
  void deallocate(void* ptr) const override;
};

}} // namespace at::cuda
