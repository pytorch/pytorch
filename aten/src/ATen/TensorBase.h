#pragma once

#include "ATen/TensorImpl.h"
#include "ATen/UndefinedTensor.h"

namespace at { namespace detail {

// TensorBase is the base class for Tensor which handles the reference counting
struct TensorBase {
  TensorBase(): TensorBase(UndefinedTensor::singleton(), false) {}
  TensorBase(TensorImpl * self, bool retain)
  : pImpl(self) {
    if (pImpl == nullptr) {
      throw std::runtime_error("TensorBase with nullptr not supported");
    }
    if(retain && pImpl != UndefinedTensor::singleton())
      pImpl->retain();
  }
  TensorBase(const TensorBase & rhs)
  : pImpl(rhs.pImpl) {
    if (pImpl != UndefinedTensor::singleton())
      pImpl->retain();
  }
  TensorBase(TensorBase && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = UndefinedTensor::singleton();
  }
  ~TensorBase() {
    if (pImpl != UndefinedTensor::singleton())
      pImpl->release();
  }
  TensorBase & operator=(TensorBase && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  TensorBase & operator=(TensorBase const & rhs) & {
      //TensorBase ctor retains original rhs.pImpl
      //then rhs.pImpl is swapped with this->pImpl
      //finally TensorBase dtor releases rhs.pImpl, which was originally this->pImpl
      TensorBase(rhs).swap(*this);
      return *this;
  }
  int64_t dim() const {
    return pImpl->dim();
  }
  void reset() {
    TensorBase().swap(*this);
  }
  void reset(TensorImpl * rhs) {
    TensorBase(rhs, true).swap(*this);
  }
  void reset(TensorImpl * rhs, bool retain) {
    TensorBase(rhs, retain).swap(*this );
  }
  void swap(TensorBase & rhs) {
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
};

}} // namespace at::detail
