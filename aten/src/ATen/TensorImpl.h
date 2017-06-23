#pragma once

#include <atomic>

#include "ATen/Type.h"
#include <iostream>
namespace at {

class Type;
class Scalar;
struct TensorImpl {
  TensorImpl(Type * type)
  : type_(type), refcount(1), is_scalar(false) {}
  Type & type() const {
    return *type_;
  }
  virtual const char * toString() const = 0;
  virtual IntList sizes() = 0;
  virtual IntList strides() = 0;
  virtual int64_t dim() = 0;
  virtual Scalar localScalar() = 0;
  void retain() {
    ++refcount;
  }
  virtual void release() {
    if(--refcount == 0) {
      delete this;
    }
  }
  virtual ~TensorImpl() {}
  friend class Type;

  // 0-dim patchup of TH requires us to have a flag marking
  // if a Tensor should be treated as 0-dim.
  // the generated wrapper manipulates this flag.
  // the setter should never be exposed in Tensor's public API
  // because eventually we would like isScalar() to just be dim() == 0;
  bool isScalar() const {
    return is_scalar;
  }
  void setScalar(bool s) {
    is_scalar = s;
  }

private:
  std::atomic<int> refcount;
  bool is_scalar;
  Type * type_;
};

}
