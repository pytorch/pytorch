#pragma once

#include <memory>
#include <stddef.h>

#include <ATen/Error.h>
#include <ATen/Retainable.h>

namespace at {

// Note [Supervisor deleter]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// TODO WRITE ME

using DeleterFnPtr = void(*)(void*);
using SupervisorPtr = std::unique_ptr<void, DeleterFnPtr>;

struct SupervisedPtr {
  // Lifetime tied to supervisor_
  void* data_;
  SupervisorPtr supervisor_;
  SupervisedPtr() : data_(nullptr), supervisor_(nullptr, nullptr) {}
  SupervisedPtr(void* data, SupervisorPtr&& supervisor)
    : data_(data), supervisor_(std::move(supervisor)) {}
  void* operator->() const { return data_; }
  void* get() const { return data_; }
};

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
