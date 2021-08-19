#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/util/Exception.h>

namespace at {
class TORCH_API OptionalTensorRef {
 public:
  OptionalTensorRef() = default;

  ~OptionalTensorRef() {
    ref_.unsafeReleaseTensorImpl();
  }

  OptionalTensorRef(const TensorBase& src)
      : ref_(Tensor::unsafe_borrow_t{}, src) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src.defined());
  }

  OptionalTensorRef(const OptionalTensorRef& rhs)
      : ref_(Tensor::unsafe_borrow_t{}, rhs.ref_) {}

  OptionalTensorRef& operator=(OptionalTensorRef rhs) {
    std::swap(ref_, rhs.ref_);
    return *this;
  }

  bool has_value() const {
    return ref_.defined();
  }

  const Tensor& getTensorRef() const & {
    return ref_;
  }

  const Tensor& operator*() const & {
    return ref_;
  }

  const Tensor* operator->() const & {
    return &ref_;
  }

  operator bool() const {
    return ref_.defined();
  }

 private:
  Tensor ref_;
};
} // namespace at
