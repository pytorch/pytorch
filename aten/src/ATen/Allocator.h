#pragma once

#include <memory>
#include <stddef.h>

#include <ATen/Device.h>
#include <ATen/core/Error.h>
#include <ATen/core/UniqueVoidPtr.h>

namespace at {

// A DataPtr is a unique pointer (with an attached deleter and some
// context for the deleter) to some memory, which also records what
// device is for its data.
//
// nullptr DataPtrs can still have a nontrivial device; this allows
// us to treat zero-size allocations uniformly with non-zero allocations.
//
class DataPtr {
private:
  detail::UniqueVoidPtr ptr_;
  Device device_;
public:
  // Choice of CPU here is arbitrary; if there's an "undefined" device
  // we could use that too
  DataPtr() : ptr_(), device_(DeviceType::CPU) {}
  DataPtr(void* data, Device device)
    : ptr_(data), device_(device) {}
  DataPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device)
    : ptr_(data, ctx, ctx_deleter), device_(device) {}
  void* operator->() const { return ptr_.get(); }
  void clear() {
    ptr_.clear();
  }
  void* get() const { return ptr_.get(); }
  void* get_context() const { return ptr_.get_context(); }
  void* release_context() { return ptr_.release_context(); }
  operator bool() const { return static_cast<bool>(ptr_); }
  template <typename T>
  T* cast_context(DeleterFnPtr expected_deleter) const {
    return ptr_.cast_context<T>(expected_deleter);
  }
  Device device() const { return device_; }
};

// NB: Device is NOT tested for here; a CUDA nullptr is as much a nullptr as a
// CPU nullptr

inline bool operator==(const at::DataPtr& dp, std::nullptr_t) noexcept { return !dp; }
inline bool operator==(std::nullptr_t, const at::DataPtr& dp) noexcept { return !dp; }
inline bool operator!=(const at::DataPtr& dp, std::nullptr_t) noexcept { return dp; }
inline bool operator!=(std::nullptr_t, const at::DataPtr& dp) noexcept { return dp; }

// Note [raw_allocate/raw_deallocate and Thrust]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Thrust's support for custom allocators requires us to write something
// like this:
//
//  class ThrustAllocator {
//    char* allocate(size_t);
//    void deallocate(char*, size_t);
//  };
//
// This is not good for our unique_ptr based allocator interface, as
// there is no way to get to the context when we free.
//
// However, in some cases the context is exactly the same as
// the data pointer.  In this case, we can support the "raw"
// allocate and deallocate interface.  This is what
// raw_deleter signifies.  By default, it returns a nullptr, which means that
// the raw interface is not implemented.  Be sure to implement it whenever
// possible, or the raw interface will incorrectly reported as unsupported,
// when it is actually possible.

struct Allocator {
  virtual ~Allocator() {}
  virtual at::DataPtr allocate(size_t n) const = 0;

  // If this returns a non nullptr, it means that allocate()
  // is guaranteed to return a unique_ptr with this deleter attached;
  // it means the rawAllocate and rawDeallocate APIs are safe to use.
  // This function MUST always return the same BoundDeleter.
  virtual DeleterFnPtr raw_deleter() const { return nullptr; }
  void* raw_allocate(size_t n) {
    auto dptr = allocate(n);
    AT_ASSERT(dptr.get() == dptr.get_context());
    return dptr.release_context();
  }
  void raw_deallocate(void* ptr) {
    auto d = raw_deleter();
    AT_ASSERT(d);
    d(ptr);
  }
};

struct AT_API InefficientStdFunctionContext {
  std::unique_ptr<void, std::function<void(void*)>> ptr_;
  InefficientStdFunctionContext(std::unique_ptr<void, std::function<void(void*)>>&& ptr)
    : ptr_(std::move(ptr)) {}
  static at::DataPtr makeDataPtr(void* ptr, const std::function<void(void*)>& deleter, Device device);
};

}  // namespace at
