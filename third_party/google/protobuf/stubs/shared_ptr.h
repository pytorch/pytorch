// Protocol Buffers - Google's data interchange format
// Copyright 2014 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// from google3/util/gtl/shared_ptr.h

#ifndef GOOGLE_PROTOBUF_STUBS_SHARED_PTR_H__
#define GOOGLE_PROTOBUF_STUBS_SHARED_PTR_H__

#include <google/protobuf/stubs/atomicops.h>

#include <algorithm>  // for swap
#include <stddef.h>
#include <memory>

namespace google {
namespace protobuf {
namespace internal {

// Alias to std::shared_ptr for any C++11 platform,
// and for any supported MSVC compiler.
#if !defined(UTIL_GTL_USE_STD_SHARED_PTR) && \
    (defined(COMPILER_MSVC) || defined(LANG_CXX11))
#define UTIL_GTL_USE_STD_SHARED_PTR 1
#endif

#if defined(UTIL_GTL_USE_STD_SHARED_PTR) && UTIL_GTL_USE_STD_SHARED_PTR

// These are transitional.  They will be going away soon.
// Please just #include <memory> and just type std::shared_ptr yourself, instead
// of relying on this file.
//
// Migration doc: http://go/std-shared-ptr-lsc
using std::enable_shared_from_this;
using std::shared_ptr;
using std::static_pointer_cast;
using std::weak_ptr;

#else  // below, UTIL_GTL_USE_STD_SHARED_PTR not set or set to 0.

// For everything else there is the google3 implementation.
inline bool RefCountDec(volatile Atomic32 *ptr) {
  return Barrier_AtomicIncrement(ptr, -1) != 0;
}

inline void RefCountInc(volatile Atomic32 *ptr) {
  NoBarrier_AtomicIncrement(ptr, 1);
}

template <typename T> class shared_ptr;
template <typename T> class weak_ptr;

// This class is an internal implementation detail for shared_ptr. If two
// shared_ptrs point to the same object, they also share a control block.
// An "empty" shared_pointer refers to NULL and also has a NULL control block.
// It contains all of the state that's needed for reference counting or any
// other kind of resource management. In this implementation the control block
// happens to consist of two atomic words, the reference count (the number
// of shared_ptrs that share ownership of the object) and the weak count
// (the number of weak_ptrs that observe the object, plus 1 if the
// refcount is nonzero).
//
// The "plus 1" is to prevent a race condition in the shared_ptr and
// weak_ptr destructors. We need to make sure the control block is
// only deleted once, so we need to make sure that at most one
// object sees the weak count decremented from 1 to 0.
class SharedPtrControlBlock {
  template <typename T> friend class shared_ptr;
  template <typename T> friend class weak_ptr;
 private:
  SharedPtrControlBlock() : refcount_(1), weak_count_(1) { }
  Atomic32 refcount_;
  Atomic32 weak_count_;
};

// Forward declaration. The class is defined below.
template <typename T> class enable_shared_from_this;

template <typename T>
class shared_ptr {
  template <typename U> friend class weak_ptr;
 public:
  typedef T element_type;

  shared_ptr() : ptr_(NULL), control_block_(NULL) {}

  explicit shared_ptr(T* ptr)
      : ptr_(ptr),
        control_block_(ptr != NULL ? new SharedPtrControlBlock : NULL) {
    // If p is non-null and T inherits from enable_shared_from_this, we
    // set up the data that shared_from_this needs.
    MaybeSetupWeakThis(ptr);
  }

  // Copy constructor: makes this object a copy of ptr, and increments
  // the reference count.
  template <typename U>
  shared_ptr(const shared_ptr<U>& ptr)
      : ptr_(NULL),
        control_block_(NULL) {
    Initialize(ptr);
  }
  // Need non-templated version to prevent the compiler-generated default
  shared_ptr(const shared_ptr<T>& ptr)
      : ptr_(NULL),
        control_block_(NULL) {
    Initialize(ptr);
  }

  // Assignment operator. Replaces the existing shared_ptr with ptr.
  // Increment ptr's reference count and decrement the one being replaced.
  template <typename U>
  shared_ptr<T>& operator=(const shared_ptr<U>& ptr) {
    if (ptr_ != ptr.ptr_) {
      shared_ptr<T> me(ptr);   // will hold our previous state to be destroyed.
      swap(me);
    }
    return *this;
  }

  // Need non-templated version to prevent the compiler-generated default
  shared_ptr<T>& operator=(const shared_ptr<T>& ptr) {
    if (ptr_ != ptr.ptr_) {
      shared_ptr<T> me(ptr);   // will hold our previous state to be destroyed.
      swap(me);
    }
    return *this;
  }

  // TODO(austern): Consider providing this constructor. The draft C++ standard
  // (20.8.10.2.1) includes it. However, it says that this constructor throws
  // a bad_weak_ptr exception when ptr is expired. Is it better to provide this
  // constructor and make it do something else, like fail with a CHECK, or to
  // leave this constructor out entirely?
  //
  // template <typename U>
  // shared_ptr(const weak_ptr<U>& ptr);

  ~shared_ptr() {
    if (ptr_ != NULL) {
      if (!RefCountDec(&control_block_->refcount_)) {
        delete ptr_;

        // weak_count_ is defined as the number of weak_ptrs that observe
        // ptr_, plus 1 if refcount_ is nonzero.
        if (!RefCountDec(&control_block_->weak_count_)) {
          delete control_block_;
        }
      }
    }
  }

  // Replaces underlying raw pointer with the one passed in.  The reference
  // count is set to one (or zero if the pointer is NULL) for the pointer
  // being passed in and decremented for the one being replaced.
  //
  // If you have a compilation error with this code, make sure you aren't
  // passing NULL, nullptr, or 0 to this function.  Call reset without an
  // argument to reset to a null ptr.
  template <typename Y>
  void reset(Y* p) {
    if (p != ptr_) {
      shared_ptr<T> tmp(p);
      tmp.swap(*this);
    }
  }

  void reset() {
    reset(static_cast<T*>(NULL));
  }

  // Exchanges the contents of this with the contents of r.  This function
  // supports more efficient swapping since it eliminates the need for a
  // temporary shared_ptr object.
  void swap(shared_ptr<T>& r) {
    using std::swap;  // http://go/using-std-swap
    swap(ptr_, r.ptr_);
    swap(control_block_, r.control_block_);
  }

  // The following function is useful for gaining access to the underlying
  // pointer when a shared_ptr remains in scope so the reference-count is
  // known to be > 0 (e.g. for parameter passing).
  T* get() const {
    return ptr_;
  }

  T& operator*() const {
    return *ptr_;
  }

  T* operator->() const {
    return ptr_;
  }

  long use_count() const {
    return control_block_ ? control_block_->refcount_ : 1;
  }

  bool unique() const {
    return use_count() == 1;
  }

 private:
  // If r is non-empty, initialize *this to share ownership with r,
  // increasing the underlying reference count.
  // If r is empty, *this remains empty.
  // Requires: this is empty, namely this->ptr_ == NULL.
  template <typename U>
  void Initialize(const shared_ptr<U>& r) {
    // This performs a static_cast on r.ptr_ to U*, which is a no-op since it
    // is already a U*. So initialization here requires that r.ptr_ is
    // implicitly convertible to T*.
    InitializeWithStaticCast<U>(r);
  }

  // Initializes *this as described in Initialize, but additionally performs a
  // static_cast from r.ptr_ (V*) to U*.
  // NOTE(gfc): We'd need a more general form to support const_pointer_cast and
  // dynamic_pointer_cast, but those operations are sufficiently discouraged
  // that supporting static_pointer_cast is sufficient.
  template <typename U, typename V>
  void InitializeWithStaticCast(const shared_ptr<V>& r) {
    if (r.control_block_ != NULL) {
      RefCountInc(&r.control_block_->refcount_);

      ptr_ = static_cast<U*>(r.ptr_);
      control_block_ = r.control_block_;
    }
  }

  // Helper function for the constructor that takes a raw pointer. If T
  // doesn't inherit from enable_shared_from_this<T> then we have nothing to
  // do, so this function is trivial and inline. The other version is declared
  // out of line, after the class definition of enable_shared_from_this.
  void MaybeSetupWeakThis(enable_shared_from_this<T>* ptr);
  void MaybeSetupWeakThis(...) { }

  T* ptr_;
  SharedPtrControlBlock* control_block_;

#ifndef SWIG
  template <typename U>
  friend class shared_ptr;

  template <typename U, typename V>
  friend shared_ptr<U> static_pointer_cast(const shared_ptr<V>& rhs);
#endif
};

// Matches the interface of std::swap as an aid to generic programming.
template <typename T> void swap(shared_ptr<T>& r, shared_ptr<T>& s) {
  r.swap(s);
}

template <typename T, typename U>
shared_ptr<T> static_pointer_cast(const shared_ptr<U>& rhs) {
  shared_ptr<T> lhs;
  lhs.template InitializeWithStaticCast<T>(rhs);
  return lhs;
}

// See comments at the top of the file for a description of why this
// class exists, and the draft C++ standard (as of July 2009 the
// latest draft is N2914) for the detailed specification.
template <typename T>
class weak_ptr {
  template <typename U> friend class weak_ptr;
 public:
  typedef T element_type;

  // Create an empty (i.e. already expired) weak_ptr.
  weak_ptr() : ptr_(NULL), control_block_(NULL) { }

  // Create a weak_ptr that observes the same object that ptr points
  // to.  Note that there is no race condition here: we know that the
  // control block can't disappear while we're looking at it because
  // it is owned by at least one shared_ptr, ptr.
  template <typename U> weak_ptr(const shared_ptr<U>& ptr) {
    CopyFrom(ptr.ptr_, ptr.control_block_);
  }

  // Copy a weak_ptr. The object it points to might disappear, but we
  // don't care: we're only working with the control block, and it can't
  // disappear while we're looking at because it's owned by at least one
  // weak_ptr, ptr.
  template <typename U> weak_ptr(const weak_ptr<U>& ptr) {
    CopyFrom(ptr.ptr_, ptr.control_block_);
  }

  // Need non-templated version to prevent default copy constructor
  weak_ptr(const weak_ptr& ptr) {
    CopyFrom(ptr.ptr_, ptr.control_block_);
  }

  // Destroy the weak_ptr. If no shared_ptr owns the control block, and if
  // we are the last weak_ptr to own it, then it can be deleted. Note that
  // weak_count_ is defined as the number of weak_ptrs sharing this control
  // block, plus 1 if there are any shared_ptrs. We therefore know that it's
  // safe to delete the control block when weak_count_ reaches 0, without
  // having to perform any additional tests.
  ~weak_ptr() {
    if (control_block_ != NULL &&
        !RefCountDec(&control_block_->weak_count_)) {
      delete control_block_;
    }
  }

  weak_ptr& operator=(const weak_ptr& ptr) {
    if (&ptr != this) {
      weak_ptr tmp(ptr);
      tmp.swap(*this);
    }
    return *this;
  }
  template <typename U> weak_ptr& operator=(const weak_ptr<U>& ptr) {
    weak_ptr tmp(ptr);
    tmp.swap(*this);
    return *this;
  }
  template <typename U> weak_ptr& operator=(const shared_ptr<U>& ptr) {
    weak_ptr tmp(ptr);
    tmp.swap(*this);
    return *this;
  }

  void swap(weak_ptr& ptr) {
    using std::swap;  // http://go/using-std-swap
    swap(ptr_, ptr.ptr_);
    swap(control_block_, ptr.control_block_);
  }

  void reset() {
    weak_ptr tmp;
    tmp.swap(*this);
  }

  // Return the number of shared_ptrs that own the object we are observing.
  // Note that this number can be 0 (if this pointer has expired).
  long use_count() const {
    return control_block_ != NULL ? control_block_->refcount_ : 0;
  }

  bool expired() const { return use_count() == 0; }

  // Return a shared_ptr that owns the object we are observing. If we
  // have expired, the shared_ptr will be empty. We have to be careful
  // about concurrency, though, since some other thread might be
  // destroying the last owning shared_ptr while we're in this
  // function.  We want to increment the refcount only if it's nonzero
  // and get the new value, and we want that whole operation to be
  // atomic.
  shared_ptr<T> lock() const {
    shared_ptr<T> result;
    if (control_block_ != NULL) {
      Atomic32 old_refcount;
      do {
        old_refcount = control_block_->refcount_;
        if (old_refcount == 0)
          break;
      } while (old_refcount !=
               NoBarrier_CompareAndSwap(
                   &control_block_->refcount_, old_refcount,
                   old_refcount + 1));
      if (old_refcount > 0) {
        result.ptr_ = ptr_;
        result.control_block_ = control_block_;
      }
    }

    return result;
  }

 private:
  void CopyFrom(T* ptr, SharedPtrControlBlock* control_block) {
    ptr_ = ptr;
    control_block_ = control_block;
    if (control_block_ != NULL)
      RefCountInc(&control_block_->weak_count_);
  }

 private:
  element_type* ptr_;
  SharedPtrControlBlock* control_block_;
};

template <typename T> void swap(weak_ptr<T>& r, weak_ptr<T>& s) {
  r.swap(s);
}

// See comments at the top of the file for a description of why this class
// exists, and section 20.8.10.5 of the draft C++ standard (as of July 2009
// the latest draft is N2914) for the detailed specification.
template <typename T>
class enable_shared_from_this {
  friend class shared_ptr<T>;
 public:
  // Precondition: there must be a shared_ptr that owns *this and that was
  // created, directly or indirectly, from a raw pointer of type T*. (The
  // latter part of the condition is technical but not quite redundant; it
  // rules out some complicated uses involving inheritance hierarchies.)
  shared_ptr<T> shared_from_this() {
    // Behavior is undefined if the precondition isn't satisfied; we choose
    // to die with a CHECK failure.
    CHECK(!weak_this_.expired()) << "No shared_ptr owns this object";
    return weak_this_.lock();
  }
  shared_ptr<const T> shared_from_this() const {
    CHECK(!weak_this_.expired()) << "No shared_ptr owns this object";
    return weak_this_.lock();
  }

 protected:
  enable_shared_from_this() { }
  enable_shared_from_this(const enable_shared_from_this& other) { }
  enable_shared_from_this& operator=(const enable_shared_from_this& other) {
    return *this;
  }
  ~enable_shared_from_this() { }

 private:
  weak_ptr<T> weak_this_;
};

// This is a helper function called by shared_ptr's constructor from a raw
// pointer. If T inherits from enable_shared_from_this<T>, it sets up
// weak_this_ so that shared_from_this works correctly. If T does not inherit
// from weak_this we get a different overload, defined inline, which does
// nothing.
template<typename T>
void shared_ptr<T>::MaybeSetupWeakThis(enable_shared_from_this<T>* ptr) {
  if (ptr) {
    CHECK(ptr->weak_this_.expired()) << "Object already owned by a shared_ptr";
    ptr->weak_this_ = *this;
  }
}

#endif  // UTIL_GTL_USE_STD_SHARED_PTR

}  // internal
}  // namespace protobuf
}  // namespace google

#endif  // GOOGLE_PROTOBUF_STUBS_SHARED_PTR_H__
