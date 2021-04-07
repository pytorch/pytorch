#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/in_place.h>

#include <cstring>
#include <type_traits>

namespace c10 {

/// A smart pointer around either a borrowed or owned T. Maintains an
/// internal raw pointer when constructed with borrowed(), with all
/// the attendant lifetime concerns.  Compare to Rust's
/// std::borrow::Cow
/// (https://doc.rust-lang.org/std/borrow/enum.Cow.html), but note
/// that it is probably not suitable for general use because C++ has
/// no borrow checking. Included here to support
/// Tensor::expect_contiguous. Importantly, requires that T is
/// pointer-sized, has alignment greater than 2, and will always have
/// a 0 in the least significant bit of its
/// representation. c10::intrusive_ptr (and types like Tensor whose
/// representation is just a c10::intrusive_ptr) meets these
/// requirements.
template <typename T>
class MaybeOwned final {
  union State {
    intptr_t borrow;
    T own;
    State() : borrow(1) {}
    ~State() {
      if (!isBorrowed()) {
        own.~T();
      }
    }
    explicit State(const T* p) : borrow((intptr_t)p | 0x1) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(((intptr_t)p & 1) == 0, "unaligned pointer used to construct MaybeOwned");
    }
    explicit State(T&& t) : own(std::move(t)) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!isBorrowed());
    }
    template <class... Args>
    explicit State(in_place_t, Args&&... args) : own(std::forward<Args>(args)...) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!isBorrowed());
    }

    State(const State& rhs) {
      if (C10_LIKELY(rhs.isBorrowed())) {
        borrow = rhs.borrow;
      } else {
        new (&own) T(rhs.own);
      }
    }

    State& operator=(const State& rhs) {
      if (C10_LIKELY(isBorrowed())) {
        if (C10_LIKELY(rhs.isBorrowed())) {
          borrow = rhs.borrow;
        } else {
          new (&own) T(rhs.own);
        }
      } else {
        if (rhs.isBorrowed()) {
          own.~T();
          borrow = rhs.borrow;
        } else {
          own = rhs.own;
        }
      }
      return *this;
    }

    State(State&& rhs) noexcept(std::is_nothrow_move_constructible<T>::value) {
      if (C10_LIKELY(rhs.isBorrowed())) {
        borrow = rhs.borrow;
      } else {
        new (&own) T(std::move(rhs.own));
      }
    }

    State& operator=(State&& rhs) noexcept(std::is_nothrow_move_assignable<T>::value) {
      if (C10_LIKELY(isBorrowed())) {
        if (C10_LIKELY(rhs.isBorrowed())) {
          borrow = rhs.borrow;
        } else {
          new (&own) T(std::move(rhs.own));
        }
      } else {
        if (rhs.isBorrowed()) {
          own.~T();
          borrow = rhs.borrow;
        } else {
          own = std::move(rhs.own);
        }
      }
      return *this;
    }

    bool isBorrowed() const noexcept {
      static_assert(sizeof(T) == sizeof(T*), "MaybeOwned assumes that T is pointer-sized");
      static_assert(alignof(T) > 1, "MaybeOwned assumes that T has alignment at least 2");
      intptr_t x;
      std::memcpy(&x, this, sizeof(x));
      return (x & 0x1) != 0;
    }

    const T* getPtr() const noexcept {
      return C10_LIKELY(isBorrowed()) ? (const T*)(borrow & ~0x1) : &own;
    }
  } u_;

  /// Don't use this; use borrowed() instead.
  explicit MaybeOwned(const T& t) : u_(&t) {}

  /// Don't use this; use owned() instead.
  explicit MaybeOwned(T&& t) noexcept(std::is_nothrow_move_constructible<T>::value)
      : u_(std::move(t)) {}

  /// Don't use this; use owned() instead.
  template <class... Args>
  explicit MaybeOwned(in_place_t, Args&&... args)
      : u_(in_place, std::forward<Args>(args)...) {}

  bool isBorrowed() const noexcept {
    return u_.isBorrowed();
  }

 public:
  explicit MaybeOwned() = default;

  // Copying a borrow yields another borrow of the original, as with a
  // T*. Copying an owned T yields another owned T for safety: no
  // chains of borrowing by default! (Note you could get that behavior
  // with MaybeOwned<T>::borrowed(*rhs) if you wanted it.)
  MaybeOwned(const MaybeOwned& rhs) = default;

  MaybeOwned& operator=(const MaybeOwned& rhs) = default;

  MaybeOwned(MaybeOwned&& rhs) noexcept(std::is_nothrow_move_constructible<T>::value) = default;

  MaybeOwned& operator=(MaybeOwned&& rhs) noexcept(std::is_nothrow_move_assignable<T>::value) = default;

  static MaybeOwned borrowed(const T& t) noexcept {
    return MaybeOwned(t);
  }

  static MaybeOwned owned(T&& t) noexcept(std::is_nothrow_move_constructible<T>::value) {
    return MaybeOwned(std::move(t));
  }

  template <class... Args>
  static MaybeOwned owned(in_place_t, Args&&... args) {
    return MaybeOwned(in_place, std::forward<Args>(args)...);
  }

  ~MaybeOwned() = default;

  const T& operator*() const {
    if (isBorrowed()) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(u_.getPtr() != nullptr);
    }
    return *u_.getPtr();
  }

  const T* operator->() const {
    if (isBorrowed()) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(u_.getPtr() != nullptr);
    }
    return u_.getPtr();
  }
};


} // namespace c10
