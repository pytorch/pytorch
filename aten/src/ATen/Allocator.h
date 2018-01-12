#pragma once

#include <memory>
#include <stddef.h>

#include "ATen/Retainable.h"

namespace at {

struct Allocator {
  virtual ~Allocator() {}
  virtual void* allocate(std::size_t n) const = 0;
  virtual void deallocate(void* ptr) const = 0;
};

namespace detail {

struct AllocatorRetainable : public Retainable {
  AllocatorRetainable(std::unique_ptr<Allocator> allocator)
    : allocator(std::move(allocator)) {}

  void* allocate(std::size_t n) {
    return allocator->allocate(n);
  }
  void deallocate(void* ptr) {
    return allocator->deallocate(ptr);
  }
private:
  std::unique_ptr<Allocator> allocator;
};

}  // namespace at::detail

}  // namespace at
