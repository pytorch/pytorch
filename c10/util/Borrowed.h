#pragma once

namespace c10 {

/// Borrowed<T> is a smart-pointer-like wrapper around a borrowed
/// instance of some type T that normally has mandatory reference
/// counting (i.e., Tensor). It provides the following interesting
/// properties:
///
/// - Can vend `const T&` and `const T*`
/// - Does not require reference count operations to create/destroy/copy (hence
/// "borrowed")
/// - Must manually be guaranteed not to outlive the underlying T it was
/// borrowed from.
///
/// You might reasonably ask at this point, "Why do I need this? I
/// have `const T&` already, which does all these things." Unlike
/// `const T&`, `Borrowed<T>` can be created without an instance of
/// `T` handy (for example, from a TensorImpl*), and then it can be used
/// to manufacture a `const T&` from there.
///
/// While it might make sense to unify Borrowed and/or BorrowedTraits
/// with MaybeOwned and/or MaybeOwnedTraits in the future, I'm keeping
/// them separate to start because MaybeOwned can work if the borrow
/// type is something other than T whereas Borrowed is only
/// interesting because it wraps a real T.

template <typename T>
struct BorrowedTraits;

template <typename T>
class Borrowed {
  using BT = BorrowedTraits<T>;
  union {
    char dummy_;
    typename BT::repr_type repr_;
  };

 public:
  Borrowed() : repr_(BT::nullRepr()) {}

  ~Borrowed() {
    // Skip destruction! We could provide a hook such as
    // BorrowedTraits::destroyRepr, but currently one is not needed.
  }

  explicit Borrowed(const T& t) : repr_(BT::copyToRepr(t)) {}

  explicit Borrowed(typename BT::raw_impl_pointer_type p)
      : repr_(BT::reprFromRawImplPointer(p)) {}

  Borrowed(const Borrowed& rhs) : repr_(BT::copyRepr(rhs.repr_)) {}

  Borrowed& operator=(const Borrowed& rhs) {
    BT::assignRepr(
        repr_,
        rhs.repr_); // NOTE: handling self-assignment is delegated to assignRepr
    return *this;
  }

  Borrowed& operator=(const T& rhs) {
    *this = Borrowed(rhs);
    return *this;
  }

  explicit operator bool() const noexcept {
    return !BT::isNullRepr(repr_);
  }

  const T& operator*() const {
    return BT::referenceFromRepr(repr_);
  };

  const T* operator->() const {
    return get();
  }

  const T* get() const {
    return BT::getImpl(repr_);
  }
};
} // namespace c10
