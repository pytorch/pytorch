#pragma once

#include <memory>
#include <stddef.h>

#include "ATen/Retainable.h"

namespace at {

struct Allocator {
  virtual ~Allocator() {}
  virtual void* allocate(size_t n) const = 0;
  virtual void deallocate(void* ptr) const = 0;
};

}  // namespace at
