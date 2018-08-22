#pragma once

#include <atomic>
#include <memory>

#include "ATen/StorageImpl.h"
#include "ATen/core/optional.h"
#include "ATen/core/TensorTypeId.h"
#include "ATen/core/TensorTypeIdRegistration.h"

struct THTensor;

namespace at {
class Scalar;
struct Type;
struct Storage;
struct Tensor;
} // namespace at

namespace at {
struct AT_API TensorImpl : public Retainable {
  TensorImpl(TensorTypeId type_id, ScalarType scalar_type, bool is_variable);
  TensorImpl(StorageImpl* storage, TensorTypeId type_id, bool is_variable);

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

  // TODO: make these protected
  // Note: storage->size() may be greater than the recorded size
  // of a tensor
  at::StorageImpl* storage_;
  int64_t storage_offset_;

  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;

  template <typename T>
  inline T * data() const {
    return storageImpl()->data<T>() + storage_offset_;
  }

  template <typename T>
  inline T * unsafe_data() const {
    return storageImpl()->unsafe_data<T>() + storage_offset_;
  }

  inline at::ScalarType scalar_type() const {
    return scalar_type_;
  }

  virtual int64_t storage_offset() const {
    return storage_offset_;
  }

  // represents that numel() == 0.
  inline bool is_empty() const {
    for (int64_t i = 0; i < dim(); ++i) {
      if (sizes()[i] == 0) {
        return true;
      }
    }
    return false;
  }

  virtual int64_t size(int64_t d) const;
  virtual int64_t stride(int64_t d) const;

  // TODO: get rid of this.
  virtual at::StorageImpl* storageImpl() const { return storage_; }

protected:
  TensorTypeId type_id_;
  // INVARIANT: When storage is non-null, this scalar type must
  // agree with the scalar type in storage
  ScalarType scalar_type_;
  bool is_variable_ = false;
  bool is_wrapped_number_ = false;

private:
  TensorImpl(StorageImpl* storage, TensorTypeId type_id, ScalarType scalar_type, bool is_variable);
};
} // namespace at
