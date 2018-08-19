#pragma once

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
//  - the underlying object stays live as long as there are any strong
//    or weak pointers to it (weakcount > 0, since strong
//    references count as a +1 to weakcount)
//
//  - underlying_object::release_resources() is called when refcount == 0
//
//  - the underlying object is destructed when weakcount == 0 (which implies
//  refcount == 0)
//
//  - Once refcount == 0, it can never again be > 0 (the transition
//    from > 0 to == 0 is monotonic)
//

struct THFinalizer {
  virtual void operator()() = 0;
  virtual ~THFinalizer() {};
};

namespace at {

struct Type;

struct AT_API StorageImpl : public Retainable {
 public:
  StorageImpl() = delete;
  virtual ~StorageImpl() {};
  StorageImpl(
      at::ScalarType scalar_type,
      ptrdiff_t size,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable);
  StorageImpl(
      at::ScalarType scalar_type,
      ptrdiff_t size,
      at::Allocator* allocator,
      bool resizable);
  StorageImpl(StorageImpl&) = delete;
  StorageImpl(const StorageImpl&) = delete;
  // NB: Don't move ref count!
  StorageImpl(StorageImpl&& other) = delete;
  StorageImpl(const StorageImpl&&) = delete;
  StorageImpl& operator=(StorageImpl&& other) = delete;

  // TODO: Rename this into th_data, and move it out of the class;
  // the real data shouldn't call th::from_type
  template <typename T>
  inline T* data() const {
    auto scalar_type_T = at::CTypeToScalarType<th::from_type<T>>::to();
    if (scalar_type_ != scalar_type_T) {
      AT_ERROR(
          "Attempt to access StorageImpl having data type ",
          at::toString(scalar_type_),
          " as data type ",
          at::toString(scalar_type_T));
    }
    return unsafe_data<T>();
  }

  template <typename T>
  inline T* unsafe_data() const {
    return static_cast<T*>(this->data_ptr_.get());
  }

  void release_resources() {
    if (finalizer_) {
      (*finalizer_)();
    }
    finalizer_ = nullptr;
    data_ptr_.clear();
  }

  void operator=(const StorageImpl&) = delete;

  virtual size_t elementSize() const {
    return at::elementSize(scalar_type_);
  }

  Type& type();

  // TODO: Rename to size() and size to size_
  ptrdiff_t size() const {
    return size_;
  };
  void set_size(ptrdiff_t size) {
    size_ = size;
  };
  bool resizable() const {
    return resizable_;
  };
  at::DataPtr& data_ptr() {
    return data_ptr_;
  };
  void set_data_ptr(at::DataPtr&& data_ptr) {
    data_ptr_ = std::move(data_ptr);
  };
  void* data() {
    return data_ptr_.get();
  };
  const void* data() const {
    return data_ptr_.get();
  };
  at::Allocator* allocator() {
    return allocator_;
  };
  at::ScalarType& scalar_type() {
    return scalar_type_;
  };
  const at::Allocator* allocator() const {
    return allocator_;
  };
  int getDevice() const {
    return data_ptr_.device().index();
  }
  void set_resizable(bool resizable) {
    resizable_ = resizable;
  }

 private:
  at::ScalarType scalar_type_;
  at::DataPtr data_ptr_;
  ptrdiff_t size_;
  bool resizable_;

 public:
  at::Allocator* allocator_;
  std::unique_ptr<THFinalizer> finalizer_;
};

namespace detail {
AT_API Backend get_backend(StorageImpl* storage_impl);
}
} // namespace at
