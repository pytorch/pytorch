#pragma once

#include "absl/types/span.h"

namespace lazy_tensors {

namespace span_internal {

// Wrappers for access to container data pointers.
template <typename C>
constexpr auto GetDataImpl(C& c, char) noexcept  // NOLINT(runtime/references)
    -> decltype(c.data()) {
  return c.data();
}

// Before C++17, std::string::data returns a const char* in all cases.
inline char* GetDataImpl(std::string& s,  // NOLINT(runtime/references)
                         int) noexcept {
  return &s[0];
}

template <typename C>
constexpr auto GetData(C& c) noexcept  // NOLINT(runtime/references)
    -> decltype(GetDataImpl(c, 0)) {
  return GetDataImpl(c, 0);
}

// Detection idioms for size() and data().
template <typename C>
using HasSize =
    std::is_integral<absl::decay_t<decltype(std::declval<C&>().size())>>;

// We want to enable conversion from vector<T*> to Span<const T* const> but
// disable conversion from vector<Derived> to Span<Base>. Here we use
// the fact that U** is convertible to Q* const* if and only if Q is the same
// type or a more cv-qualified version of U.  We also decay the result type of
// data() to avoid problems with classes which have a member function data()
// which returns a reference.
template <typename T, typename C>
using HasData =
    std::is_convertible<absl::decay_t<decltype(GetData(std::declval<C&>()))>*,
                        T* const*>;

}  // namespace span_internal

template <typename T>
class Span {
 private:
  // Used to determine whether a Span can be constructed from a container of
  // type C.
  template <typename C>
  using EnableIfConvertibleFrom =
      typename std::enable_if<span_internal::HasData<T, C>::value &&
                              span_internal::HasSize<C>::value>::type;

  // Used to SFINAE-enable a function when the slice elements are const.
  template <typename U>
  using EnableIfConstView =
      typename std::enable_if<std::is_const<T>::value, U>::type;

 public:
  using value_type = absl::remove_cv_t<T>;
  using pointer = T*;

  constexpr Span() noexcept = default;

  // Implicit reference constructor for a read-only `Span<const T>` type
  template <typename V, typename = EnableIfConvertibleFrom<V>,
            typename = EnableIfConstView<V>>
  constexpr Span(const V& v) noexcept : impl_(v) {}

  Span(std::initializer_list<value_type> v) noexcept : impl_(v) {}

  constexpr pointer data() const noexcept { return impl_.data(); }

  constexpr auto size() const noexcept { return impl_.size(); }

  constexpr bool empty() const noexcept { return impl_.empty(); }

  constexpr T& operator[](size_t i) const noexcept { return impl_[i]; }

  constexpr T& at(size_t i) const { return impl_.at(i); }

  constexpr T& front() const noexcept { return impl_.front(); }

  constexpr T& back() const noexcept { return impl_.back(); }

  constexpr auto begin() const noexcept { return impl_.begin(); }

  constexpr auto end() const noexcept { return impl_.end(); }

  constexpr auto rbegin() const noexcept { return impl_.rbegin(); }

  constexpr auto rend() const noexcept { return impl_.rend(); }

  static bool equals(Span<T> a, Span<T> b) { return a.impl_ == b.impl_; }

 private:
  absl::Span<T> impl_;
};

template <typename T>
bool operator==(Span<T> a, Span<T> b) {
  return Span<T>::equals(a, b);
}

template <typename T>
bool operator!=(Span<T> a, Span<T> b) {
  return !(a == b);
}

}  // namespace lazy_tensors
