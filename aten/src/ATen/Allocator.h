#pragma once

#include <memory>
#include <stddef.h>

#include <ATen/Error.h>
#include <ATen/Retainable.h>

namespace at {

// Note [Separated Allocator and Deleter]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Why are Allocator and Deleter put in separate classes?  The key is that
// an allocator may allocate a distinct context for every deleter.  This is
// especially important upon reallocation: if we do not allocate a new
// context, the contexts of the new and old data can clobber each other.
// Imagine the following sequence of events:
//
//  1. Storage has some data and a BoundAllocatorDeleter associated with it.
//     The context in this case is an owning reference to "IOBuf", an
//     enclosing struct for the data.
//
//  2. A resize occurs on storage.  We call BoundAllocatorDeleter to
//     allocate some new memory to store the resized data.  To allocate
//     this new memory, we must allocate a new IOBuf.  But how can
//     we update the context to replace the old reference with
//     the new one?  Disaster!
//
// Previously, this case was worked around by directly supporting realloc()
// in the deleter.  But this is bad for different reasons (it assumes the
// allocator knows how to copy data; not a safe assumption, since allocators
// don't know what data is actually contained within them.)

struct Deleter {
  virtual ~Deleter() {}
  virtual void deallocate(void* ctx, void* ptr) const = 0;
};

// WARNING: BoundDeleter may LEAK ctx_ if you never actually call it on
// the pointer it's supposed to delete; e.g., ctx_'s lifetime may be the
// same as the pointer, and invocation of deallocate() is necessary to
// ensure that deallocation of ctx_ happens at the same time ptr is
// deallocated.
struct BoundDeleter final {
  at::Deleter* deleter_;
  void* ctx_;
  BoundDeleter() : deleter_(nullptr), ctx_(nullptr) {}
  BoundDeleter(at::Deleter* deleter, void* ctx) : deleter_(deleter), ctx_(ctx) {
    if (!deleter_) { AT_ASSERT(ctx == nullptr); }
  }
  void operator()(void* ptr) {
    if (deleter_) { deleter_->deallocate(ctx_, ptr); }
  }
  operator bool() {
    return static_cast<bool>(deleter_);
  }
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
  virtual std::unique_ptr<void, BoundDeleter> allocate(size_t n) const = 0;

  // If this returns a callable deleter, it means that allocate()
  // is guaranteed to return a unique_ptr with this deleter attached;
  // it means the rawAllocate and rawDeallocate APIs are safe to use.
  // This function MUST always return the same BoundDeleter.
  virtual BoundDeleter maybeGlobalBoundDeleter() const { return {}; }
  void* raw_allocate(size_t n) {
    AT_ASSERT(maybeGlobalBoundDeleter());
    return allocate(n).release();
  }
  void raw_deallocate(void* ptr) {
    auto d = maybeGlobalBoundDeleter();
    AT_ASSERT(d);
    d(ptr);
  }

  // There's a slight inefficiency here.  We are unable to inline
  // across maybeGlobalBoundDeleter, because it is virtual, but
  // this inlining might be quite profitable because the deleter
  // is likely to be quite simple so we might be able to completely
  // eliminate the BoundDeleter struct.  However, if we virtualized
  // raw_allocate/raw_deallocate, in principle they could be generated
  // for every subclass, with maybeGlobalBoundDeleter inlined (because
  // all subclasses of Allocator ought to be final).  Unfortunately,
  // doing this is a bit unreadable, so we leave this optimization
  // to future work.
};

// An inefficient deleter that stashes a pointer to a dynamically allocated
// std::function in the context.  This is a "use-at-last-resort" deleter.
struct InefficientStdFunctionDeleter : public at::Deleter {
  void deallocate(void* ctx, void* ptr) const override {
    auto* fnptr = static_cast<std::function<void(void*)>*>(ctx);
    (*fnptr)(ptr);
    delete fnptr;
  }
  static BoundDeleter make(const std::function<void(void*)> & deleter) {
    return {&singleton_, new std::function<void(void*)>(deleter)};
  }
private:
  static InefficientStdFunctionDeleter singleton_;
};

}  // namespace at
