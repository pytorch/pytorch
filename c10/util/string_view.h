#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <c10/macros/Macros.h>

namespace c10 {

template <class CharT, class Traits = std::char_traits<CharT>>
class basic_string_view final : public std::basic_string_view<CharT, Traits> {
 public:
  using const_pointer = const CharT*;
  using std::basic_string_view<CharT, Traits>::basic_string_view;

  constexpr basic_string_view() noexcept = default;
  constexpr basic_string_view(const basic_string_view&) noexcept = default;
  constexpr basic_string_view& operator=(
      const basic_string_view& rhs) noexcept = default;
  constexpr basic_string_view(basic_string_view&&) noexcept = default;
  constexpr basic_string_view& operator=(basic_string_view&& rhs) noexcept =
      default;

  constexpr basic_string_view(
      const ::std::basic_string_view<CharT, Traits>& str) noexcept
      : basic_string_view(str.data(), str.size()) {}
  constexpr basic_string_view(const ::std::basic_string<CharT>& str) noexcept
      : basic_string_view(str.data(), str.size()) {}

  constexpr operator ::std::basic_string_view<CharT, Traits>() const noexcept {
    return ::std::basic_string_view<CharT>(this->data(), this->size());
  }

  constexpr operator ::std::basic_string<CharT>() const {
    return ::std::basic_string<CharT>(this->data(), this->size());
  }

  constexpr auto data() const noexcept {
    return std::basic_string_view<CharT, Traits>::data();
  }

  constexpr auto size() const noexcept {
    return std::basic_string_view<CharT, Traits>::size();
  }

  constexpr bool starts_with(basic_string_view prefix) const noexcept {
    return (prefix.size() > this->size())
        ? false
        : prefix == this->substr(0, prefix.size());
  }

  constexpr bool starts_with(CharT prefix) const noexcept {
    return !this->empty() && prefix == this->front();
  }

  constexpr bool starts_with(const_pointer prefix) const noexcept {
    return starts_with(basic_string_view(prefix));
  }

  constexpr bool ends_with(basic_string_view suffix) const noexcept {
    return (suffix.size() > this->size())
        ? false
        : suffix == this->substr(this->size() - suffix.size(), suffix.size());
  }

  constexpr bool ends_with(CharT suffix) const noexcept {
    return !this->empty() && suffix == this->back();
  }

  constexpr bool ends_with(const_pointer suffix) const noexcept {
    return ends_with(basic_string_view(suffix));
  }
};

template <class CharT, class Traits>
constexpr bool operator==(
    basic_string_view<CharT, Traits> lhs,
    basic_string_view<CharT, Traits> rhs) noexcept {
  return static_cast<std::basic_string_view<CharT, Traits>>(lhs) ==
      static_cast<std::basic_string_view<CharT, Traits>>(rhs);
}
template <class CharT, class Traits>
constexpr bool operator!=(
    basic_string_view<CharT, Traits> lhs,
    basic_string_view<CharT, Traits> rhs) noexcept {
  return static_cast<std::basic_string_view<CharT, Traits>>(lhs) !=
      static_cast<std::basic_string_view<CharT, Traits>>(rhs);
}
template <class CharT, class Traits>
constexpr bool operator<(
    basic_string_view<CharT, Traits> lhs,
    basic_string_view<CharT, Traits> rhs) noexcept {
  return static_cast<std::basic_string_view<CharT, Traits>>(lhs) <
      static_cast<std::basic_string_view<CharT, Traits>>(rhs);
}
template <class CharT, class Traits>
constexpr bool operator<=(
    basic_string_view<CharT, Traits> lhs,
    basic_string_view<CharT, Traits> rhs) noexcept {
  return static_cast<std::basic_string_view<CharT, Traits>>(lhs) <=
      static_cast<std::basic_string_view<CharT, Traits>>(rhs);
}
template <class CharT, class Traits>
constexpr bool operator>(
    basic_string_view<CharT, Traits> lhs,
    basic_string_view<CharT, Traits> rhs) noexcept {
  return static_cast<std::basic_string_view<CharT, Traits>>(lhs) >
      static_cast<std::basic_string_view<CharT, Traits>>(rhs);
}
template <class CharT, class Traits>
constexpr bool operator>=(
    basic_string_view<CharT, Traits> lhs,
    basic_string_view<CharT, Traits> rhs) noexcept {
  return static_cast<std::basic_string_view<CharT, Traits>>(lhs) >=
      static_cast<std::basic_string_view<CharT, Traits>>(rhs);
}

template <class CharT>
constexpr void swap(
    basic_string_view<CharT>& lhs,
    basic_string_view<CharT>& rhs) noexcept {
  lhs.swap(rhs);
}
using string_view = basic_string_view<char>;

} // namespace c10

namespace std {
template <class CharT>
struct hash<::c10::basic_string_view<CharT>> {
  size_t operator()(::c10::basic_string_view<CharT> x) const {
    // Forwards to std::string_view
    using std_string_view_type = ::std::basic_string_view<CharT>;
    return ::std::hash<std_string_view_type>{}(
        std_string_view_type(x.data(), x.size()));
  }
};
} // namespace std
