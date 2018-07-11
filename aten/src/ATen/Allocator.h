#pragma once

#include <memory>
#include <stddef.h>

#include <ATen/Error.h>
#include <ATen/Retainable.h>

namespace at {

// Note [Supervisor deleter]
// ~~~~~~~~~~~~~~~~~~~~~~~~~
// SupervisorPtr solves a common problem for allocators of tensor data, which
// is that the data pointer (e.g., float*) which you are interested in, is not
// the same as the metadata pointer (e.g., DLManagedTensor) which you need
// to actually deallocate the data.  Under a conventional deleter design, you
// have to store extra context in the deleter (the metadata pointer) so that
// you can actually delete the right thing.  Implementing this with standard
// C++ is somewhat error-prone: if you use a std::unique_ptr to manage tensors,
// the deleter will not be called if the data pointer is nullptr, which can
// cause a leak if the metadata pointer is non-null (and the deleter is
// responsible for freeing both the data pointer and the metadata pointer).
//
// We take a different approach.  The "metadata supervisor" situation is common
// enough that we have organized our deleter strategy entirely around it:
// instead of trying to make the deleter for the data pointer handle all the
// heavy lifting, the data pointer is *non-owning*, and instead there is a
// (type-erased) supervisor pointer which actually handles deletion.  For simple
// cases, the supervisor pointer is the same as the data pointer, but if
// there is some extra metadata, the supervisor pointer points there.
//
// There is something of a pattern to writing these; check THAllocator.{h,cpp}
// for some examples.

using DeleterFnPtr = void(*)(void*);
using SupervisorPtr = std::unique_ptr<void, DeleterFnPtr>;

// Does not delete anything
AT_API void deleteNothing(void*);
// Mints a SupervisorPtr that doesn't do anything.  You must
// use this, not a nullptr, when you want a view.
AT_API SupervisorPtr nonOwningSupervisorPtr();

struct SupervisedPtr {
  // Lifetime tied to supervisor_
  void* data_;
  SupervisorPtr supervisor_;
  SupervisedPtr() : data_(nullptr), supervisor_(nonOwningSupervisorPtr()) {}
  SupervisedPtr(void* data, SupervisorPtr&& supervisor)
    : data_(data), supervisor_(std::move(supervisor)) {}
  void* operator->() const { return data_; }
  void* get() const { return data_; }
  operator bool() { return data_ || supervisor_; }
  SupervisedPtr& operator=( std::nullptr_t ) noexcept { data_ = nullptr; supervisor_ = nullptr; return *this; }
};

inline bool operator==(const at::SupervisedPtr& sp, std::nullptr_t) noexcept {
  return sp.data_ == nullptr && sp.supervisor_ == nullptr;
}
inline bool operator==(std::nullptr_t, const at::SupervisedPtr& sp) noexcept {
  return sp.data_ == nullptr && sp.supervisor_ == nullptr;
}
inline bool operator!=(const at::SupervisedPtr& sp, std::nullptr_t) noexcept {
  return sp.data_ != nullptr || sp.supervisor_ != nullptr;
}
inline bool operator!=(std::nullptr_t, const at::SupervisedPtr& sp) noexcept {
  return sp.data_ != nullptr || sp.supervisor_ != nullptr;
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
// there is no way to get the deleter from our allocator to the
// deletion site in Thrust.  Indeed, if every pointer is getting a
// *fresh* deleter, we truly would have no choice except to maintain
// a map from pointers to deleters.  This is bad.
//
// So, we observe that not *all* deleters actually have lots of
// different deleters; some of them actually always return the same
// deleter every time.  In this case, we can support the "raw"
// allocate and deallocate interface.  This is what
// maybeGlobalBoundDeleter signifies.  By default, it returns the
// default (empty) BoundDeleter, which means that the raw interface
// is not implemented.  Be sure to implement it whenever possible.

struct Allocator {
  virtual ~Allocator() {}
  virtual at::SupervisedPtr allocate(size_t n) const = 0;

  // If this returns a non nullptr, it means that allocate()
  // is guaranteed to return a unique_ptr with this deleter attached;
  // it means the rawAllocate and rawDeallocate APIs are safe to use.
  // This function MUST always return the same BoundDeleter.
  virtual DeleterFnPtr raw_deleter() const { return nullptr; }
  void* raw_allocate(size_t n) {
    auto sptr = allocate(n);
    AT_ASSERT(sptr.data_ == sptr.supervisor_.get());
    return sptr.supervisor_.release();
  }
  void raw_deallocate(void* ptr) {
    auto d = raw_deleter();
    AT_ASSERT(d);
    d(ptr);
  }
};

struct AT_API InefficientStdFunctionSupervisor {
  std::unique_ptr<void, std::function<void(void*)>> ptr_;
  InefficientStdFunctionSupervisor(std::unique_ptr<void, std::function<void(void*)>>&& ptr)
    : ptr_(std::move(ptr)) {}
};

AT_API at::SupervisedPtr
makeInefficientStdFunctionSupervisedPtr(void* ptr, const std::function<void(void*)>& deleter);

}  // namespace at
