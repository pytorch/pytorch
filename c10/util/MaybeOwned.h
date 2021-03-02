#pragma once

#include <c10/util/in_place.h>

namespace c10 {

/// A smart pointer around either a borrowed or owned T. Maintains an
/// internal raw pointer when constructed with borrowed(), with all
/// the attendant lifetime concerns.  Compare to Rust's
/// std::borrow::Cow
/// (https://doc.rust-lang.org/std/borrow/enum.Cow.html).
template <typename T>
class TORCH_API MaybeOwned {
  bool isBorrowed_;
  union {
    const T *borrow_;
    T own_;
  };
  explicit MaybeOwned(const T& t) : isBorrowed_(true), borrow_(&t) {}
  explicit MaybeOwned(T&& t) : isBorrowed_(false), own_(std::move(t)) {}
  template <class... Args>
  explicit MaybeOwned(in_place_t, Args&&... args)
  : isBorrowed_(false)
  , own_(std::forward<Args>(args)...) {}

 public:
  static MaybeOwned borrowed(const T& t) {
    return MaybeOwned(t);
  }

  static MaybeOwned owned(T&& t) {
    return MaybeOwned(std::move(t));
  }

  template <class... Args>
  static MaybeOwned owned(in_place_t, Args&&... args) {
    return MaybeOwned(in_place, std::forward<Args>(args)...);
  }

  ~MaybeOwned() {
    if (!isBorrowed_) {
      own_.~T();
    }
  }

  const T& operator*() const {
    return isBorrowed_ ? *borrow_ : own_;
  }

  const T* operator->() const {
    return isBorrowed_ ? borrow_ : &own_;
  }
};


} // namespace c10
