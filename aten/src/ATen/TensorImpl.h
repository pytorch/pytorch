#pragma once

#include <atomic>
#include <memory>

#include "ATen/Retainable.h"
#include "ATen/ScalarType.h"

namespace at {
class Scalar;
struct Type;
struct Storage;
struct Tensor;
} // namespace at

namespace at {
struct TensorImpl : public Retainable {
  explicit TensorImpl(Type * type)
  : is_scalar(false), type_(type) {}

  Type & type() const {
    return *type_;
  }
  virtual const char * toString() const = 0;
  virtual IntList sizes() const = 0;
  virtual IntList strides() const = 0;
  virtual int64_t dim() const = 0;
  virtual Scalar localScalar() = 0;
  virtual void * unsafeGetTH(bool retain) = 0;
  virtual std::unique_ptr<Storage> storage() = 0;
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

  // ~~~~~ Autograd API ~~~~~
  // Some methods below are defined in TensorImpl.cpp because Tensor is an
  // incomplete type.

  AT_API virtual void set_requires_grad(bool requires_grad) {
    AT_ERROR("set_requires_grad is not implemented for Tensor");
  }
  AT_API virtual bool requires_grad() const {
    AT_ERROR("requires_grad is not implemented for Tensor");
  }

  AT_API virtual Tensor& grad();
  AT_API virtual const Tensor& grad() const;

  AT_API virtual Tensor detach() const;
  AT_API virtual void detach_() {
    AT_ERROR("detach_ is not implemented for Tensor");
  }

  AT_API virtual void set_data(Tensor new_data);

protected:
  bool is_scalar;
  Type * type_;
};
} // namespace at
