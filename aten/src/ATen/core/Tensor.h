#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/util/Exception.h>

namespace at {
class TORCH_API OptionalTensorRef {
 public:
  OptionalTensorRef() {}

  ~OptionalTensorRef() {
    ref_.unsafeReleaseTensorImpl();
  }

  OptionalTensorRef(const Tensor& src)
      : ref_(c10::intrusive_ptr<TensorImpl>(
            src.unsafeGetTensorImpl(),
            c10::raw::DontIncreaseRefcount{})) {}

  OptionalTensorRef(const OptionalTensorRef& rhs)
      : OptionalTensorRef(rhs.ref_) {}

  OptionalTensorRef& operator=(const OptionalTensorRef& rhs) {
    // Need to call unsafeReleaseTensorImpl on ref_ since we are reassigning it
    // (which does not call the destructor).
    ref_.unsafeReleaseTensorImpl();
    ref_ = Tensor(c10::intrusive_ptr<TensorImpl>(
        rhs.ref_.unsafeGetTensorImpl(), c10::raw::DontIncreaseRefcount{}));
    return *this;
  }

  // Implementation of move constructor is the same as copy constructor.
  // rhs has a non-owning reference, so it doesn't need to be cleared.
  OptionalTensorRef(OptionalTensorRef&& rhs) : OptionalTensorRef(rhs.ref_) {}

  OptionalTensorRef& operator=(OptionalTensorRef&& rhs) {
    // Implementation of move assignment operator is the same as copy assignment
    // operator. rhs has a non-owning reference, so it doesn't need to be
    // cleared.
    *this = rhs;
    return *this;
  }

  bool has_value() const {
    return ref_.defined();
  }

  const Tensor& getTensorRef() const & {
    return ref_;
  }

  operator bool() const {
    return ref_.defined();
  }

 private:
  Tensor ref_;
};
} // namespace at
