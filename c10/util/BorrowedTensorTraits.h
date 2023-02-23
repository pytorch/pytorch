#pragma once

#include <c10/core/TensorImpl.h>
#include <cstring>

namespace c10 {
// This traits class is very much logically part of the private
// implementation of TensorBase. It pulls memcpy shenanigans that rely
// on knowing how TensorBase (and intrusive_ptr) are represented and
// work. However, the safety of this is guarded with tests that should
// be running under ASAN.
template <typename TensorType>
struct BorrowedTraits {
  using repr_type = TensorType;
  using raw_impl_pointer_type = TensorImpl*;

  static TensorType nullRepr() {
    // intrusive_ptr(std::nullptr_t) sets the internal representation
    // to NullType::singleton() instead. We want actual nullptr bit
    // pattern. We will not dereference it or run the
    // TensorImpl/intrusive_ptr destructor.
    TensorType t;
    std::nullptr_t nul = nullptr;
    static_assert(sizeof(t) == sizeof(nul));
    std::memcpy(&t, &nul, sizeof(nul));
    return t;
  }

  static TensorType copyToRepr(const TensorType& t) {
    static_assert(sizeof(TensorType) == sizeof(void*));
    // Blind pointer copy. Cannot use the unsafe_borrow_t constructor
    // because rhs might be nullptr, which intrusive_ptr::reclaim does
    // not like.
    TensorType result;
    std::memcpy(&result, &t, sizeof(result));
    return result;
  }

  static TensorType reprFromRawImplPointer(TensorImpl* p) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(p != nullptr);
    return TensorType(
        c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(p));
  }

  static TensorType copyRepr(const TensorType& t) {
    return copyToRepr(t);
  }

  static void assignRepr(TensorType& lhs, const TensorType& rhs) {
    static_assert(sizeof(TensorType) == sizeof(void*));
    // Blind pointer copy. Cannot use the unsafe_borrow_t constructor
    // because rhs might be nullptr, which intrusive_ptr::reclaim does
    // not like.
    std::memcpy(&lhs, &rhs, sizeof(lhs));
  }

  static bool isNullRepr(const TensorType& t) {
    return t.unsafeGetTensorImpl() == nullptr;
  }

  static const TensorType& referenceFromRepr(const TensorType& t) {
    return t;
  }

  static const TensorType* getImpl(const TensorType& t) {
    return &t;
  }
};
} // namespace c10
