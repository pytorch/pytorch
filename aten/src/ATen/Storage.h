#pragma once

#include <ATen/Allocator.h>

#include <ATen/ScalarType.h>
#include <ATen/ScalarTypeUtils.h>
#include <TH/THTypeConversion.hpp>
#include <atomic>

#include "ATen/Scalar.h"
#include <ATen/Half.h>

#include <ATen/Config.h>

// Legacy for compatability with TH!
struct THFinalizer {
  virtual void operator()() = 0;
  virtual ~THFinalizer() {};
};

namespace at {

struct Type;

struct Storage {
  Storage() {}

  Storage(at::Backend, at::ScalarType, int64_t, at::DataPtr, at::Allocator*, bool);
  Storage(at::Backend, at::ScalarType, int64_t, at::Allocator*, bool);

  at::Backend backend;
  at::ScalarType scalar_type;
  at::DataPtr data_ptr;
  int64_t size_;
  std::atomic<int> refcount;
  std::atomic<int> weakcount;
  bool resizable_;
  at::Allocator* allocator;
  std::unique_ptr<THFinalizer> finalizer;
  struct Storage* view;

  Storage(Storage&) = delete;
  Storage(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage(const Storage&&) = delete;
  void operator=(const Storage&) = delete;

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

  size_t elementSize() const {
    return at::elementSize(scalar_type);
  }
  size_t size() const {
    return size_;
  };
  void* data() {
    return data_ptr.get();
  };
  const void* data() const {
    return data_ptr.get();
  };
  void retain() {
    ++refcount;
  }
  Type& type() const;
  int getDevice() const {
    return data_ptr.device().index();
  }
  void set_resizable(bool resizable) {
    resizable_ = resizable;
  }

};

} // namespace at
