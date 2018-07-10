#pragma once

#include <memory>
#include <stddef.h>

#include <ATen/Error.h>
#include <ATen/Retainable.h>

namespace at {

// Note [Separated Allocator and Deleter]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// A Deleter knows how to delete a void* pointer, freeing it back to the
// system.  Every storage needs a deleter, so that we know how to free
// the memory.
//
// An Allocator, given a size, knows how to allocate memory of that size.
// Generally, you don't care too much about how a given piece of memory
// is allocated, but if you need to *reallocate* some memory, it's good
// to know how to reallocate something "in the same way", and that means
// you have to know what the allocator is.
//
// Below, Allocator and Deleter are split into two separate classes.  You
// might wonder, why is that?  In the common case, an allocator is in
// one-to-one correspondence with a deleter, in the same way malloc() is
// paired with free().
//
// However, there is a major exception to this case: when we write
// Allocators/Deleters for "externally" managed memory.  In this case,
// we may need an extra, externally provided pointer to some enclosing
// struct if we want to free this memory, and this pointer is *different* for
// every allocated void* pointer.
//
// To see what can go wrong, let's suppose that we had put Allocator and Deleter
// together with the context, in a single "Allocator" class:
//
//    struct Allocator {
//        void* ctx_;
//        Allocator* allocator_;
//        Deleter* deleter_;
//    }
//
// Here, ctx_ stores the pointer to the enclosing struct.  Imagine the following
// sequence of events:
//
//  1. Storage has some data and a Allocator associated with it.
//     The context in the Allocator is an owning reference to "IOBuf", an
//     enclosing struct for the data; the function to free an IOBuf
//     is freeIOBuf(IOBuf*), NOT freeIOBuf(void*).
//
//  2. A resize occurs on storage.  We call Allocator to
//     allocate some new memory to store the resized data.  To allocate
//     this new memory, we must allocate a new IOBuf.  Now we have a
//     problem: the classic API for an allocator is void*(void* ctx, size_t size).
//     Where are we going to put the freshly allocated IOBuf?  We can't write it
//     into the context directly, because that will clobber the old
//     IOBuf (which we need to keep live until we copy the data out.)
//
// Instead, the allocator should *return* a new context for the deleter,
// and this is what we have done below.  (We have further simplified matters
// by saying that an allocator never has a context; we haven't seen any cases
// where this is necessary, but it is a possible future extension).
//
// By the way, previously, this case was worked around by directly supporting
// realloc() in the deleter.  But this is bad for different reasons (it assumes
// the allocator knows how to copy data; not a safe assumption, since allocators
// don't know the type of the data is actually contained within them; if the
// data has a non-trivial copy constructor, there's no way to do a resize safely
// in this case.)

struct Deleter {
  virtual ~Deleter() {}
  virtual void deallocate(void* ctx, void* ptr) const = 0;
};

// WARNING: A common pattern when writing BoundDeleter is to set things
// up so that the lifetime of ctx_ is the same as pointer.  In this case,
// you will LEAK ctx_ if you never actually call it on the pointer it's
// supposed to delete.  Use the deleter that comes with a pointer to
// deallocate it; don't do it some out-of-band way!
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
