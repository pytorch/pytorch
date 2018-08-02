#pragma once

#include <ATen/Scalar.h>

#include <ATen/Allocator.h>
#include <ATen/ScalarType.h>
#include <ATen/ScalarTypeUtils.h>
#include <ATen/Retainable.h>
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
//  - StorageImpl stays live as long as there are any strong
//    or weak pointers to it (weakcount > 0, since strong
//    references count as a +1 to weakcount)
//
//  - finalizers are called and data_ptr is deallocated when refcount == 0
//
//  - Once refcount == 0, it can never again be > 0 (the transition
//    from > 0 to == 0 is monotonic)
//
//  - When you access StorageImpl via a weak pointer, you must
//    atomically increment the use count, if it is greater than 0.
//    If it is not, you must report that the storage is dead.
//

struct THFinalizer {
  virtual void operator()() = 0;
  virtual ~THFinalizer() {};
};

namespace at {

struct Type;

struct AT_API StorageImpl : public Retainable {

  StorageImpl() = delete;
  virtual ~StorageImpl() {};
  StorageImpl(at::ScalarType, ptrdiff_t, at::DataPtr, at::Allocator*, bool);
  StorageImpl(at::ScalarType, ptrdiff_t, at::Allocator*, bool);
  at::ScalarType scalar_type;
  at::DataPtr data_ptr;
  ptrdiff_t size;
  bool resizable;
  at::Allocator* allocator;
  std::unique_ptr<THFinalizer> finalizer;
  StorageImpl(StorageImpl&) = delete;
  StorageImpl(const StorageImpl&) = delete;
  StorageImpl(StorageImpl&&) = delete;
  StorageImpl(const StorageImpl&&) = delete;

  // TODO: Rename this into th_data, and move it out of the class;
  // the real data shouldn't call th::from_type
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

  void release_resources() {
    if (finalizer) {
      (*finalizer)();
    }
    finalizer = nullptr;
    data_ptr.clear();
  }

  void operator=(const StorageImpl&) = delete;

  virtual size_t elementSize() const {
    return at::elementSize(scalar_type);
  }

  Type& type();

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

  int getDevice() const {
    return data_ptr.device().index();
  }
  void set_resizable(bool resizable_) {
    resizable = resizable_;
  }
};

} // namespace at
