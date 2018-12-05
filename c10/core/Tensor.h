#pragma once

#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/macros/Macros.h>

namespace c10 {

class C10Tensor final {
private:
  using TensorImplPtr = intrusive_ptr<TensorImpl, UndefinedTensorImpl>;
public:
  explicit C10Tensor(TensorImplPtr impl) noexcept;

  C10Tensor(const C10Tensor&) = default;
  C10Tensor(C10Tensor&&) noexcept = default;
  C10Tensor& operator=(const C10Tensor&) = default;
  C10Tensor& operator=(C10Tensor&&) noexcept = default;

  const TensorImplPtr &impl() const & noexcept;
  TensorImplPtr impl() && noexcept;

  TensorTypeId type_id() const;

private:
  TensorImplPtr impl_;
};

inline C10Tensor::C10Tensor(TensorImplPtr impl) noexcept
: impl_(std::move(impl)) {}

inline const C10Tensor::TensorImplPtr &C10Tensor::impl() const & noexcept {
  return impl_;
}

inline C10Tensor::TensorImplPtr C10Tensor::impl() && noexcept {
  return std::move(impl_);
}

inline TensorTypeId C10Tensor::type_id() const {
  return impl_->type_id();
}

} // namespace c10
