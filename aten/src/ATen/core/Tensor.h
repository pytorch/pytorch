#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/util/Exception.h>

namespace at {
class TORCH_API OptionalTensorRef {
 public:
  OptionalTensorRef() {}
  OptionalTensorRef(const Tensor& src)
      : ref_(c10::intrusive_ptr<TensorImpl>(
            src.unsafeGetTensorImpl(),
            c10::raw::DontIncreaseRefcount{})) {
    TORCH_INTERNAL_ASSERT(src.defined());
  }

  ~OptionalTensorRef() {
    ref_.unsafeReleaseTensorImpl();
  }

  bool has_value() const {
    return ref_.defined();
  }

  const Tensor& getTensorRef() const {
    return ref_;
  }

  operator bool() const {
    return ref_.defined();
  }

 private:
  Tensor ref_;
};
} // namespace at
