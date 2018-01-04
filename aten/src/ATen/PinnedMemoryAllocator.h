#pragma once

#include "Allocator.h"

namespace at {

struct PinnedMemoryAllocator final : public Allocator {
  void* allocate(std::size_t n) const override;
  void deallocate(void* ptr) const override;
};

}
