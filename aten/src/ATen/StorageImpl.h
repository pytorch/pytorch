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
//  - THStorageImpl stays live as long as there are any strong
//    or weak pointers to it (weakcount > 0, since strong
//    references count as a +1 to weakcount)
//
//  - finalizers are called and data_ptr is deallocated when refcount == 0
//
//  - Once refcount == 0, it can never again be > 0 (the transition
//    from > 0 to == 0 is monotonic)
//
//  - When you access THStorageImpl via a weak pointer, you must
//    atomically increment the use count, if it is greater than 0.
//    If it is not, you must report that the storage is dead.
//

struct THFinalizer {
  virtual void operator()() = 0;
  virtual ~THFinalizer() {};
};

namespace at {

struct Type;


// OPtions: Use retainable
// Use strorage ptr == unique ptr <custom destructor calls release on storage>
//  - custom destructor will just decrease refcount
// Could rename this to StorageImplImpl and then create StorageImpl that maintains StorageImplImpl and decrease refcount


// StorageImpl/StorageImplImpl thing
// Inherit from retainable
// StorageImpl has deleter, ; assignment and move deleted, pretty much like StorageImpl and THStorageImpl before
// - - don't have to deleted unique ptr - will use intrusive and retainable


// Details:
// Forward the constructor just like before
// Deconstructor just calls free on StorageImpl
// Create a StorageImpl free function

struct StorageImpl {

  StorageImpl() = delete;
  virtual ~StorageImpl() {};
  StorageImpl(at::ScalarType, ptrdiff_t, at::DataPtr, at::Allocator*, char);
  StorageImpl(at::ScalarType, ptrdiff_t, at::Allocator*, char);
  at::ScalarType scalar_type;
  at::DataPtr data_ptr;
  ptrdiff_t size;
  std::atomic<int> refcount;
  std::atomic<int> weakcount;
  char flag;
  at::Allocator* allocator;
  std::unique_ptr<THFinalizer> finalizer;
  StorageImpl(StorageImpl&) = delete;
  StorageImpl(const StorageImpl&) = delete;
  StorageImpl(StorageImpl&&) = delete;
  StorageImpl(const StorageImpl&&) = delete;

  template <typename T>
  inline T* data() const {
    auto scalar_type_T = at::CTypeToScalarType<th::from_type<T>>::to();
    if (scalar_type != scalar_type_T) {
      AT_ERROR(
          "Attempt to access StorageImpl having data type ",
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

  void operator=(const StorageImpl&) = delete;

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
