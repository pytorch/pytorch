#pragma once

#include "ATen/TensorImpl.h"

namespace at { namespace detail {

// TensorBase is the base class for Tensor which handles the reference counting
struct TensorBase {
  TensorBase()
  : pImpl(nullptr) {}
  TensorBase(TensorImpl * self, bool retain)
  : pImpl(self) {
    if(pImpl != nullptr && retain)
      pImpl->retain();
  }
  TensorBase(const TensorBase & rhs)
  : pImpl(rhs.pImpl) {
    if(pImpl != nullptr)
      pImpl->retain();
  }
  TensorBase(TensorBase && rhs) noexcept
  : pImpl(rhs.pImpl) {
    rhs.pImpl = nullptr;
  }
  ~TensorBase() {
    if(pImpl != nullptr)
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
  void swap(TensorBase & rhs) {
    TensorImpl * tmp = pImpl;
    pImpl = rhs.pImpl;
    rhs.pImpl = tmp;
  }
  TensorImpl * get() const {
    return pImpl;
  }

  friend struct Type;

  //TODO(zach): sort out friend structes
public:
  TensorImpl * pImpl;
};

}} // namespace at::detail
