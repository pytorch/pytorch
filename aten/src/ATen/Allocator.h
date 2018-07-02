#pragma once

#include <memory>
#include <stddef.h>

#include <ATen/Error.h>
#include <ATen/Retainable.h>

namespace at {

struct Allocator {
  virtual ~Allocator() {}
  virtual void* allocate(void* ctx, size_t n) const = 0;
  virtual void deallocate(void* ctx, void* ptr) const = 0;
};


struct StorageDeleterAllocator : public Allocator {
  void* allocate(void* ctx, size_t n) const override {
    AT_ERROR("Cannot reallocate externally provided memory");
  }
  void deallocate(void* ctx, void* ptr) const override {
    auto* fnptr = static_cast<std::function<void(void*)>*>(ctx);
    (*fnptr)(ptr);
    delete fnptr;
  }
};
Allocator* getStorageDeleterAllocator();

}  // namespace at
