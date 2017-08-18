#pragma once

#include <atomic>

#include "ATen/Type.h"
#include <iostream>
namespace at {

struct Type;
class Scalar;
struct TensorImpl {
  explicit TensorImpl(Type * type)
  :  refcount(1), is_scalar(false), type_(type) {}
  Type & type() const {
    return *type_;
  }
  virtual const char * toString() const = 0;
  virtual IntList sizes() = 0;
  virtual IntList strides() = 0;
  virtual int64_t dim() = 0;
  virtual Scalar localScalar() = 0;
  virtual void assign_(Scalar s) = 0;
  virtual void * unsafeGetTH(bool retain) = 0;
  void retain() {
    ++refcount;
  }
  virtual void release() {
    if(--refcount == 0) {
      delete this;
    }
  }
  virtual ~TensorImpl() {}
  friend struct Type;

  // 0-dim patchup of TH requires us to have a flag marking
  // if a Tensor should be treated as 0-dim.
  // the generated wrapper manipulates this flag.
  // the setter should never be exposed in Tensor's public API
  // because eventually we would like isScalar() to just be dim() == 0;
  bool isScalar() const {
    return is_scalar;
  }
  // this is called by the generated wrapper code when there are conditions
  // when this output tensor should be a scalar. e.g. when all inputs
  // to a function 'add' were scalars, then condition_when_scalar == true.
  // we also prevent this from getting marked as a scalar if it is not
  // the right shape afterall.
  TensorImpl* maybeScalar(bool condition_when_scalar) {
    is_scalar = false; //force dim() to tell the truth for TH
    is_scalar = condition_when_scalar && dim() == 1 && sizes()[0] == 1;
    return this;
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
