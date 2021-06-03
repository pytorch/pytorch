#pragma once

#include <c10/util/in_place.h>

namespace c10 {

// TODO: document ExclusivelyOwnedTraits and settle on names (and
// signatures?) of operations. In the meantime, see examples in
// TensorBody.h and intrusive_ptr.h.
// REVIEW: would anyone like to argue that it doesn't make sense to
// have this traits class, and instead we should just have two
// explicit specializations of ExclusivelyOwned? There would be some
// small amount of code duplication, but perhaps it would be easier to
// understand?
template <typename T>
struct ExclusivelyOwnedTraits;

/// ExclusivelyOwned is a smart-pointer-like wrapper around an
/// exclusively-owned instance of some type T that normally has
/// mandatory reference counting (currently Tensor or
/// c10::intrusive_ptr). If you have an isolated piece of code that
/// knows that it has sole ownership of an object of one of these
/// types (i.e., because you created it directly or using a factory
/// function) and that object will not escape from that isolated piece
/// of code, then moving the object into an ExclusivelyOwned will
/// avoid an atomic reference count decrement at destruction time.
///
/// If you directly create the Tensor/intrusive_ptr in the first
/// place, you can use the in_place constructor of ExclusivelyOwned to
/// additionally avoid doing any stores to initialize the refcount &
/// weakcount. (Do note, however, that in this case you should
/// probably just use std::unique_ptr instead of intrusive_ptr if applicable.)
template <typename T>
class ExclusivelyOwned {
  using EOT = ExclusivelyOwnedTraits<T>;
  union {
    char dummy_;
    typename ExclusivelyOwnedTraits<T>::repr_type repr_;
  };

 public:
  ExclusivelyOwned() : repr_(EOT::nullRepr()) {}

  explicit ExclusivelyOwned(T&& t) : repr_(EOT::moveToRepr(std::move(t))) {}

  template <class... Args>
  explicit ExclusivelyOwned(in_place_t, Args&&... args)
      : repr_(EOT::createInPlace(std::forward<Args>(args)...)) {}

  ExclusivelyOwned(const ExclusivelyOwned&) = delete;

  ExclusivelyOwned(ExclusivelyOwned&& rhs) noexcept
      : repr_(std::move(rhs.repr_)) {
    rhs.repr_ = EOT::nullRepr();
  }

  ExclusivelyOwned& operator=(ExclusivelyOwned&& rhs) noexcept {
    EOT::destroyOwned(repr_);
    repr_ = std::move(rhs.repr_);
    rhs.repr_ = EOT::nullRepr();
    return *this;
  }

  ~ExclusivelyOwned() {
    EOT::destroyOwned(repr_);
    // End the lifetime of repr_ without executing its dtor, since we
    // already did specialized destruction for the exclusively-owned
    // case in destroyOwned!
    dummy_ = '\0';
  }

  // We don't provide this because it would require us to be able to
  // differentiate an owned-but-empty T from a lack of T. This is
  // particularly problematic for Tensor, which wants to use an
  // undefined Tensor as its null state.
  explicit operator bool() const noexcept = delete;

  operator T() && {
    return take();
  }

  // NOTE: the equivalent operation on MaybeOwned is a moving
  // operator*. For ExclusivelyOwned, take() and operator*() may well
  // have different return types (e.g., for intrusive_ptr, take()
  // returns c10::intrusive_ptr<T> whereas operator* returns T&), so
  // they are different functions.
  T take() && {
    return EOT::take(repr_);
  }

  typename EOT::pointer_type operator->() const {
    return get();
  }

  typename EOT::pointer_type get() const {
    return EOT::getImpl(repr_);
  }

  std::remove_pointer_t<typename EOT::pointer_type>& operator*() const {
    return *get();
  }

};

} // namespace c10
