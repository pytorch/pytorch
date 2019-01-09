#pragma once

#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/macros/Macros.h>

namespace c10 {

/**
 * This is a minimal Tensor class for use in c10 code.
 * The plan on record is to eventually merge at::Tensor and caffe2::Tensor
 * and move that merged class to c10, replacing this one.
 *
 * At time of writing this, we couldn't do that yet, because their APIs are
 * not clean enough to make it in c10 and because they have dependencies we want
 * to avoid, for example at::Tensor depends on at::Type.
 */
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
