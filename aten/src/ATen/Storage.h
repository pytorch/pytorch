#pragma once

#include <ATen/Scalar.h>

#include <ATen/Allocator.h>
#include <ATen/ScalarType.h>
#include <ATen/ScalarTypeUtils.h>
#include <TH/THTypeConversion.hpp>
#include <atomic>

// Note [Weak references for intrusive refcounting]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Here's the scheme:
//
//  - refcount == number of strong references to the object
//    weakcount == number of weak references to the object,
//      plus one more if refcount > 0
//
//  - THStorage stays live as long as there are any strong
//    or weak pointers to it (weakcount > 0, since strong
//    references count as a +1 to weakcount)
//
//  - finalizers are called and data_ptr is deallocated when refcount == 0
//
//  - Once refcount == 0, it can never again be > 0 (the transition
//    from > 0 to == 0 is monotonic)
//
//  - When you access THStorage via a weak pointer, you must
//    atomically increment the use count, if it is greater than 0.
//    If it is not, you must report that the storage is dead.
//

struct THFinalizer {
  virtual void operator()() = 0;
  virtual ~THFinalizer() {};
};

namespace at {

struct Type;

struct Storage {

  Storage() = delete;
  virtual ~Storage() {};
  Storage(at::ScalarType, ptrdiff_t, at::DataPtr, at::Allocator*, char);
  Storage(at::ScalarType, ptrdiff_t, at::Allocator*, char);
  at::ScalarType scalar_type;
  at::DataPtr data_ptr;
  ptrdiff_t size;
  std::atomic<int> refcount;
  std::atomic<int> weakcount;
  char flag;
  at::Allocator* allocator;
  std::unique_ptr<THFinalizer> finalizer;
  Storage(Storage&) = delete;
  Storage(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage(const Storage&&) = delete;

  template <typename T>
  inline T* data() const {
    auto scalar_type_T = at::CTypeToScalarType<th::from_type<T>>::to();
    if (scalar_type != scalar_type_T) {
      AT_ERROR(
          "Attempt to access Storage having data type ",
          at::toString(scalar_type),
          " as data type ",
          at::toString(scalar_type_T));
    }
    return unsafe_data<T>();
  }

  template <typename T>
  inline T* unsafe_data() const {
    return static_cast<T*>(this->data_ptr.get());
  }
  static const char REFCOUNTED = 1;
  static const char RESIZABLE = 2;

  void operator=(const Storage&) = delete;

  virtual size_t elementSize() const {
    return at::elementSize(scalar_type);
  }
  //TODO: Rename to size() and size to size_
  size_t get_size() const { 
    return size;
  };
  void* data() {
    return data_ptr.get();
  };
  const void* data() const {
    return data_ptr.get();
  };
  void* unsafeGetTH(bool retain_) {
    if (retain_ && flag & REFCOUNTED) {
      ++refcount;
    }
    return this;
  }
  void retain() {
    if (flag & REFCOUNTED) {
      ++refcount;
    }
  }
  Type& type() const;

  int getDevice() const {
    return data_ptr.device().index();
  }
  void clear_flag(char flag_) {
    flag &= ~flag_;
  }
};

} // namespace at
