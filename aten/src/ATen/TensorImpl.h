#pragma once

#include <atomic>
#include <memory>

#include "ATen/Retainable.h"
#include "ATen/ScalarType.h"
#include "ATen/core/optional.h"

struct THTensor;

namespace at {
class Scalar;
struct Type;
struct Storage;
struct Tensor;
} // namespace at

namespace at {
struct AT_API TensorImpl : public Retainable {
  explicit TensorImpl(Backend backend, ScalarType scalar_type, THTensor * tensor, bool is_variable)
  : backend_(backend), scalar_type_(scalar_type), is_variable_(is_variable), tensor(tensor) {}
  TensorImpl(Backend backend, ScalarType scalar_type);

  virtual ~TensorImpl();

  virtual void release_resources() override;

  // The implementation of this method will have to be hoisted out and
  // hooked in, so that Caffe2 doesn't need to know about Context
  // TODO: This really really needs to be inlined.
  Type & type() const;

  const char * toString() const;
  virtual IntList sizes() const;
  virtual IntList strides() const;
  virtual int64_t dim() const;
  virtual void * unsafeGetTH(bool retain);
  virtual std::unique_ptr<Storage> storage();
  friend struct Type;

  int64_t numel() {
    int64_t n = 1;
    for (auto s : sizes()) {
      n *= s;
    }
    return n;
  }

  // this is called by the generated wrapper code when there are conditions
  // when this output tensor should be zero dimensional. e.g. when all inputs
  // to a function 'add' were zero dimensional, then condition_when_zero_dim == true.
  // we also prevent this from getting marked as a zero dim tensor if it is not
  // the right shape afterall.
  virtual TensorImpl* maybe_zero_dim(bool condition_when_zero_dim);

  // True if a tensor was auto-wrapped from a C++ or Python number.
  // Wrapped numbers do not participate in the result type computation for
  // mixed-type operations if there are any Tensors that are not wrapped
  // numbers. Otherwise, they behave like their non-wrapped equivalents.
  // See [Result type computation] in TensorIterator.h.
  bool is_wrapped_number() const {
    return is_wrapped_number_;
  }
  void set_wrapped_number(bool value) {
    AT_ASSERT(dim() == 0);
    is_wrapped_number_ = value;
  }

  // ~~~~~ Autograd API ~~~~~
  // Some methods below are defined in TensorImpl.cpp because Tensor is an
  // incomplete type.

  virtual void set_requires_grad(bool requires_grad) {
    AT_ERROR("set_requires_grad is not implemented for Tensor");
  }
  virtual bool requires_grad() const {
    AT_ERROR("requires_grad is not implemented for Tensor");
  }

  virtual Tensor& grad();
  virtual const Tensor& grad() const;

  virtual Tensor detach() const;
  virtual void detach_() {
    AT_ERROR("detach_ is not implemented for Tensor");
  }

  virtual void backward(
      at::optional<Tensor> gradient,
      bool keep_graph,
      bool create_graph);

  virtual void set_data(Tensor new_data);

protected:
  Backend backend_;
  // INVARIANT: When storage is non-null, this scalar type must
  // agree with the scalar type in storage
  ScalarType scalar_type_;
  bool is_variable_ = false;
  bool is_wrapped_number_ = false;
public:
  THTensor * tensor;
};
} // namespace at
