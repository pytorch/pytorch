#pragma once

#include <stddef.h>
#include <memory>

#include <c10/core/Device.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/Exception.h>

namespace c10 {

// A DataPtr is a unique pointer (with an attached deleter and some
// context for the deleter) to some memory, which also records what
// device is for its data.
//
// nullptr DataPtrs can still have a nontrivial device; this allows
// us to treat zero-size allocations uniformly with non-zero allocations.
//
class C10_API DataPtr {
 private:
  c10::detail::UniqueVoidPtr ptr_;
  Device device_;

 public:
  // Choice of CPU here is arbitrary; if there's an "undefined" device
  // we could use that too
  DataPtr() : ptr_(), device_(DeviceType::CPU) {}
  DataPtr(void* data, Device device) : ptr_(data), device_(device) {}
  DataPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device)
      : ptr_(data, ctx, ctx_deleter), device_(device) {}
  void* operator->() const {
    return ptr_.get();
  }
  void clear() {
    ptr_.clear();
  }
  void* get() const {
    return ptr_.get();
  }
  void* get_context() const {
    return ptr_.get_context();
  }
  void* release_context() {
    return ptr_.release_context();
  }
  std::unique_ptr<void, DeleterFnPtr>&& move_context() {
    return ptr_.move_context();
  }
  operator bool() const {
    return static_cast<bool>(ptr_);
  }
  template <typename T>
  T* cast_context(DeleterFnPtr expected_deleter) const {
    return ptr_.cast_context<T>(expected_deleter);
  }
  DeleterFnPtr get_deleter() const {
    return ptr_.get_deleter();
  }
  /**
   * Compare the deleter in a DataPtr to expected_deleter.
   * If it matches, replace the deleter with new_deleter
   * and return true; otherwise, does nothing and returns
   * false.
   *
   * In general, it is not safe to unconditionally set the
   * deleter on a DataPtr, because you don't know what
   * the deleter is, and thus will have a hard time properly
   * disposing of the deleter without storing the original
   * deleter (this is difficult to do, because DeleterFnPtr
   * is not a closure, and because the context on DataPtr is
   * only a single word, you generally don't have enough
   * space to store both the original deleter and its context).
   * However, in some cases, you know /exactly/ what the deleter
   * is, and you have a new deleter that manually wraps
   * the old one.  In this case, you can safely swap the deleter
   * after asserting that the deleters line up.
   *
   * What are the requirements on new_deleter?  It must still
   * properly dispose of the void* pointer passed in as its argument,
   * where void* is whatever the context of the original deleter
   * is.  So in general, you expect the new deleter to look something
   * like this:
   *
   *      [](void* ptr) {
   *        some_new_stuff(ptr);
   *        get_orig_allocator()->raw_deleter(ptr);
   *      }
   *
   * Note that it won't work to close over the original
   * allocator; you don't have enough space to do that!  Also,
   * it's unsafe to assume that the passed in pointer in
   * question is the memory pointer in question; it might not
   * be; be sure to read the source code of the Allocator
   * in question to confirm this.
   */
  C10_NODISCARD bool compare_exchange_deleter(DeleterFnPtr expected_deleter, DeleterFnPtr new_deleter) {
    return ptr_.compare_exchange_deleter(expected_deleter, new_deleter);
  }
  Device device() const {
    return device_;
  }
  // Unsafely mutates the device on a DataPtr.  Under normal use,
  // you should never actually need to call this function.
  // We need this for the implementation of the hack detailed
  // in Note [Masquerading as CUDA]
  void unsafe_set_device(Device device) {
    device_ = device;
  }
};

// NB: Device is NOT tested for here; a CUDA nullptr is as much a nullptr as a
// CPU nullptr

inline bool operator==(const DataPtr& dp, std::nullptr_t) noexcept {
  return !dp;
}
inline bool operator==(std::nullptr_t, const DataPtr& dp) noexcept {
  return !dp;
}
inline bool operator!=(const DataPtr& dp, std::nullptr_t) noexcept {
  return dp;
}
inline bool operator!=(std::nullptr_t, const DataPtr& dp) noexcept {
  return dp;
}

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

struct C10_API Allocator {
  virtual ~Allocator() = default;

  virtual DataPtr allocate(size_t n) const = 0;

  // If this returns a non nullptr, it means that allocate()
  // is guaranteed to return a unique_ptr with this deleter attached;
  // it means the rawAllocate and rawDeallocate APIs are safe to use.
  // This function MUST always return the same BoundDeleter.
  virtual DeleterFnPtr raw_deleter() const {
    return nullptr;
  }
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

// This context is used to generate DataPtr which have arbitrary
// std::function deleters associated with them.  In some user facing
// functions, we give a (user-friendly) interface for constructing
// tensors from external data which take an arbitrary std::function
// deleter.  Grep for InefficientStdFunctionContext to find these
// occurrences.
//
// This context is inefficient because we have to do a dynamic
// allocation InefficientStdFunctionContext, on top of the dynamic
// allocation which is implied by std::function itself.
struct C10_API InefficientStdFunctionContext {
  std::unique_ptr<void, std::function<void(void*)>> ptr_;
  InefficientStdFunctionContext(
      std::unique_ptr<void, std::function<void(void*)>>&& ptr)
      : ptr_(std::move(ptr)) {}
  static DataPtr makeDataPtr(
      void* ptr,
      const std::function<void(void*)>& deleter,
      Device device);
};

/** Set the allocator for DeviceType `t`. The passed in allocator pointer is
 *  expected to have static lifetime; this function does NOT take ownership
 *  of the raw pointer. (The reason for this is to prevent existing pointers
 *  to an allocator of a particular device from being invalidated when
 *  SetAllocator is called.)
 *
 *  Also note that this is not thread-safe, and we assume this function will
 *  only be called during initialization.
 */
C10_API void SetAllocator(DeviceType t, Allocator* alloc);
C10_API Allocator* GetAllocator(const DeviceType& t);

template <DeviceType t>
struct AllocatorRegisterer {
  explicit AllocatorRegisterer(Allocator* alloc) {
    SetAllocator(t, alloc);
  }
};

#define REGISTER_ALLOCATOR(t, f)                    \
  namespace {                                       \
  static AllocatorRegisterer<t> g_allocator_d(f); \
  }

} // namespace c10
