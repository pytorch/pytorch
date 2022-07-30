#pragma once

#include <c10/core/TensorImpl.h>

#include <utility>

namespace c10 {
// Shared ExclusivelyOwnedTraits implementation between caffe2::Tensor and
// at::TensorBase.
template <typename TensorType>
struct ExclusivelyOwnedTensorTraits {
  using repr_type = TensorType;
  using pointer_type = TensorType*;
  using const_pointer_type = const TensorType*;

  static repr_type nullRepr() {
    return TensorType();
  }

  template <class... Args>
  static repr_type createInPlace(Args&&... args) {
    return TensorType(std::forward<Args>(args)...);
  }

  static repr_type moveToRepr(TensorType&& x) {
    return std::move(x);
  }

  static void destroyOwned(TensorType& x) {
    TensorImpl* const toDestroy = x.unsafeReleaseTensorImpl();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        toDestroy != nullptr, "Tensor somehow got null TensorImpl?");
    // May be 0 because UndefinedTensorImpl doesn't get its refcount
    // incremented.
    const bool isUndefined = toDestroy == UndefinedTensorImpl::singleton();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        toDestroy->refcount_ == 1 || (toDestroy->refcount_ == 0 && isUndefined),
        "ExclusivelyOwned<Tensor> destroyed with isUndefined ",
        isUndefined,
        " and refcount ",
        toDestroy->refcount_,
        ", expected 1 or, if isUndefined, 0!");
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        toDestroy->weakcount_ == 1 ||
            (toDestroy->weakcount_ == 0 &&
             toDestroy == UndefinedTensorImpl::singleton()),
        "ExclusivelyOwned<Tensor> destroyed with isUndefined ",
        isUndefined,
        " and weakcount ",
        toDestroy->weakcount_,
        ", expected 1 or, if isUndefined, 0!");
    if (!isUndefined) {
#ifndef NDEBUG
      // Needed to pass the debug assertions in ~intrusive_ptr_target.
      toDestroy->refcount_ = 0;
      toDestroy->weakcount_ = 0;
#endif
      delete toDestroy;
    }
  }

  static TensorType take(TensorType& x) {
    return std::move(x);
  }

  static pointer_type getImpl(repr_type& x) {
    return &x;
  }

  static const_pointer_type getImpl(const repr_type& x) {
    return &x;
  }
};
} // namespace c10
