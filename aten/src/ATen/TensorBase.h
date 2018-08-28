#pragma once

#include "ATen/TensorImpl.h"
#include "ATen/UndefinedTensor.h"
#include "ATen/core/Error.h"

namespace at { namespace detail {

// TensorBase is the base class for Tensor.
// TODO: Eliminate this, once we remove TensorBase from Scalar.  At
// the moment it's only used to break an include cycle for Scalar
struct TensorBase {
  TensorBase() {}
  TensorBase(TensorImpl * tensor_impl, bool retain) : tensor_impl_(c10::intrusive_ptr<TensorImpl, UndefinedTensor>::reclaim(tensor_impl)) {
    if (tensor_impl == nullptr) {
      throw std::runtime_error("TensorBaseImpl with nullptr not supported");
    }
    if (retain && tensor_impl != UndefinedTensor::singleton()) {
      c10::raw::intrusive_ptr::incref(tensor_impl);
    }
  }
  TensorBase(c10::intrusive_ptr<TensorImpl, UndefinedTensor>&& ptr) : tensor_impl_(std::move(ptr)) {}
  TensorBase(const c10::intrusive_ptr<TensorImpl, UndefinedTensor>& ptr) : tensor_impl_(ptr) {}

  int64_t dim() const {
    return tensor_impl_->dim();
  }

  TensorImpl * unsafeGetTensorImpl() const {
    return tensor_impl_.get();
  }
  TensorImpl * unsafeReleaseTensorImpl() {
    return tensor_impl_.release();
  }
  const c10::intrusive_ptr<TensorImpl, UndefinedTensor>& getIntrusivePtr() const {
    return tensor_impl_;
  }

  bool defined() const {
    return tensor_impl_;
  }

  void reset() {
    tensor_impl_.reset();
  }

  friend struct WeakTensor;

protected:
  c10::intrusive_ptr<TensorImpl, UndefinedTensor> tensor_impl_;
};

}} // namespace at::detail
