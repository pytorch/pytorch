#pragma once

#include "ATen/TensorImpl.h"
#include "ATen/UndefinedTensor.h"

namespace at { namespace detail {

// TensorBaseImpl is the base class for Tensor which handles the reference counting
template<bool is_strong>
struct TensorBaseImpl {
  TensorBaseImpl(): TensorBaseImpl(UndefinedTensor::singleton(), false) {}
  TensorBaseImpl(TensorImpl * self, bool should_retain)
  : pImpl(self) {
    if (pImpl == nullptr) {
      throw std::runtime_error("TensorBaseImpl with nullptr not supported");
    }
    if(should_retain && pImpl != UndefinedTensor::singleton()) {
      retain();
    }
  }
  TensorBaseImpl(const TensorBaseImpl & rhs)
  : pImpl(rhs.pImpl) {
    if (pImpl != UndefinedTensor::singleton()) {
      retain();
    }
  }
  TensorBaseImpl(TensorBaseImpl && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = UndefinedTensor::singleton();
  }
  ~TensorBaseImpl() {
    if (pImpl != UndefinedTensor::singleton()) {
      release();
    }
  }
  TensorBaseImpl & operator=(TensorBaseImpl && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  TensorBaseImpl & operator=(TensorBaseImpl const & rhs) & {
    //TensorBaseImpl ctor retains original rhs.pImpl
    //then rhs.pImpl is swapped with this->pImpl
    //finally TensorBaseImpl dtor releases rhs.pImpl, which was originally this->pImpl
    TensorBaseImpl(rhs).swap(*this);
    return *this;
  }
  int64_t dim() const {
    if (is_strong) {
      return pImpl->dim();
    } else {
      AT_ERROR("Can't call dim() on a WeakTensor");
    }
  }
  void reset() {
    TensorBaseImpl().swap(*this);
  }
  void reset(TensorImpl * rhs) {
    TensorBaseImpl(rhs, true).swap(*this);
  }
  void reset(TensorImpl * rhs, bool should_retain) {
    TensorBaseImpl(rhs, should_retain).swap(*this );
  }
  void swap(TensorBaseImpl & rhs) {
    TensorImpl * tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }
  TensorImpl * get() const {
    return pImpl;
  }
  TensorImpl * detach() {
    TensorImpl * ret = pImpl;
    pImpl = UndefinedTensor::singleton();
    return ret;
  }

  bool defined() const {
    return pImpl != UndefinedTensor::singleton();
  }

  friend struct Type;

  //TODO(zach): sort out friend structes
public:
  TensorImpl * pImpl;

private:
  void retain() {
    if (is_strong) {
      pImpl->retain();
    } else {
      pImpl->weak_retain();
    }
  }

  void release() {
    if (is_strong) {
      pImpl->release();
    } else {
      pImpl->weak_release();
    }
  }
};

using TensorBase = TensorBaseImpl<true>;
using WeakTensorBase = TensorBaseImpl<false>;

}} // namespace at::detail
