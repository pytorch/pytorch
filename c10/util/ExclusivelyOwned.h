#pragma once

#include <utility>

namespace c10 {

// See example implementation in TensorBase.h and TensorBody.h.
// Synopsis:
//
// repr_type -- type to use to store an owned T in ExclusivelyOwned.
//
// pointer_type -- pointer-esque type to return from
// ExclusivelyOwned's get() and operator*() methods.
//
// const_pointer_type -- similar to pointer_type, used for the const methods.
//
// static repr_type nullRepr() -- return a null instance of repr_type.
//
// template <class... Args>
// static repr_type createInPlace(Args&&... args) -- used by the in-place
// ExclusivelyOwned constructor.
//
// static repr_type moveToRepr(T&& x) -- move the given x into an
// instance of repr_type. used by the ExclusivelyOwned(T&&)
// constructor.
//
// static void destroyOwned(repr_type x) -- free memory for a
// known-exclusively-owned instance of x. Replaces calling repr_type's
// destructor. Being able to implement this more efficiently than
// repr_type's destructor is the main reason to use ExclusivelyOwned
// for a type.
//
// static T take(repr_type&) -- move out of the given repr_type into an owned T.
//
// static pointer_type getImpl(const repr_type&) -- return a pointer
// to the given repr_type. May take repr_type by value if that is more
// efficient.
template <typename T>
struct ExclusivelyOwnedTraits;

/// ExclusivelyOwned is a smart-pointer-like wrapper around an
/// exclusively-owned instance of some type T that normally has
/// mandatory reference counting (currently just Tensor). If you have
/// an isolated piece of code that knows that it has sole ownership of
/// an object of one of these types (i.e., because you created it
/// directly or using a factory function) and that object will not
/// escape from that isolated piece of code, then moving the object
/// into an ExclusivelyOwned will avoid an atomic reference count
/// decrement at destruction time.
///
/// If you directly create the Tensor in the first
/// place, you can use the in_place constructor of ExclusivelyOwned to
/// additionally avoid doing any stores to initialize the refcount &
/// weakcount.
template <typename T>
class ExclusivelyOwned {
  using EOT = ExclusivelyOwnedTraits<T>;
  typename ExclusivelyOwnedTraits<T>::repr_type repr_;

 public:
  ExclusivelyOwned() : repr_(EOT::nullRepr()) {}

  explicit ExclusivelyOwned(T&& t) : repr_(EOT::moveToRepr(std::move(t))) {}

  template <class... Args>
  explicit ExclusivelyOwned(std::in_place_t, Args&&... args)
      : repr_(EOT::createInPlace(std::forward<Args>(args)...)) {}

  ExclusivelyOwned(const ExclusivelyOwned&) = delete;

  ExclusivelyOwned(ExclusivelyOwned&& rhs) noexcept
      : repr_(std::move(rhs.repr_)) {
    rhs.repr_ = EOT::nullRepr();
  }

  ExclusivelyOwned& operator=(const ExclusivelyOwned&) = delete;

  ExclusivelyOwned& operator=(ExclusivelyOwned&& rhs) noexcept {
    EOT::destroyOwned(repr_);
    repr_ = std::move(rhs.repr_);
    rhs.repr_ = EOT::nullRepr();
    return *this;
  }

  ExclusivelyOwned& operator=(T&& rhs) noexcept {
    EOT::destroyOwned(repr_);
    repr_ = EOT::moveToRepr(std::move(rhs));
    return *this;
  }

  ~ExclusivelyOwned() {
    EOT::destroyOwned(repr_);
    // Don't bother to call the destructor of repr_, since we already
    // did specialized destruction for the exclusively-owned case in
    // destroyOwned!
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
  // have different return types, so they are different functions.
  T take() && {
    return EOT::take(repr_);
  }

  typename EOT::const_pointer_type operator->() const {
    return get();
  }

  typename EOT::const_pointer_type get() const {
    return EOT::getImpl(repr_);
  }

  typename EOT::pointer_type operator->() {
    return get();
  }

  typename EOT::pointer_type get() {
    return EOT::getImpl(repr_);
  }

  std::remove_pointer_t<typename EOT::const_pointer_type>& operator*() const {
    return *get();
  }

  std::remove_pointer_t<typename EOT::pointer_type>& operator*() {
    return *get();
  }
};

} // namespace c10
