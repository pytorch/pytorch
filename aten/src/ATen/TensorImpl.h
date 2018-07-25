#pragma once

#include <atomic>
#include <memory>

#include "ATen/Retainable.h"
#include "ATen/ScalarType.h"
#include "ATen/optional.h"

struct THTensor;

namespace at {
class Scalar;
struct Type;
struct Storage;
struct Tensor;
} // namespace at

namespace at {
struct AT_API TensorImpl : public Retainable {
  explicit TensorImpl(Type * type, THTensor * tensor)
  : type_(type), tensor(tensor) {}

  virtual ~TensorImpl();

  virtual void release_resources() override;

  Type & type() const {
    return *type_;
  }
  const char * toString() const;
  virtual IntList sizes() const;
  virtual IntList strides() const;
  virtual int64_t dim() const;
  virtual void * unsafeGetTH(bool retain);
  virtual std::unique_ptr<Storage> storage() = 0;
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
  Type * type_;
public:
  THTensor * tensor;
};
} // namespace at
