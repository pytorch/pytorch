#pragma once

#include <algorithm>
#include <functional>
#include <ostream>
#include <string>
#include <string_view>

#include <c10/macros/Macros.h>

namespace c10 {

/**
 * Thin wrapper over std::basic_string_view that customizes substr()
 * to avoid throwing std::out_of_range so it can be called from CUDA
 * __device__ code (which does not support C++ exceptions). It clamps
 * out-of-range positions silently.
 *
 * Most call sites should prefer `c10::string_view`, which is just
 * `std::string_view`. This class exists for the device-callable case
 * above.
 */
template <class CharT>
class basic_string_view final : public std::basic_string_view<CharT> {
 public:
  using Base = std::basic_string_view<CharT>;
  using typename Base::size_type;

  using Base::Base;
  // `using Base::Base` does not bring Base's copy/move ctors.
  /* implicit */ constexpr basic_string_view(Base sv) noexcept : Base(sv) {}

  [[nodiscard]] constexpr basic_string_view substr(
      size_type pos = 0,
      size_type count = Base::npos) const {
#if !defined(__CUDA_ARCH__)
    return basic_string_view(Base::substr(pos, count));
#else
    const size_type p = std::min(pos, Base::size());
    return basic_string_view(
        Base::data() + p, std::min(count, Base::size() - p));
#endif
  }

  [[nodiscard]] constexpr explicit operator std::basic_string<CharT>() const {
    return std::basic_string<CharT>(static_cast<const Base&>(*this));
  }
};

template <class CharT>
inline std::basic_ostream<CharT>& operator<<(
    std::basic_ostream<CharT>& stream,
    basic_string_view<CharT> sv) {
  return stream << static_cast<std::basic_string_view<CharT>>(sv);
}

using string_view = std::string_view;
using c10_string_view = basic_string_view<char>;

// NOTE: In C++20, this function should be replaced by string_view.starts_with
[[nodiscard]] constexpr bool starts_with(
    const std::string_view s,
    const std::string_view prefix) noexcept {
  return (prefix.size() > s.size()) ? false
                                    : prefix == s.substr(0, prefix.size());
}

// NOTE: In C++20, this function should be replaced by string_view.starts_with
[[nodiscard]] constexpr bool starts_with(
    const std::string_view s,
    const char prefix) noexcept {
  return !s.empty() && prefix == s.front();
}

// NOTE: In C++20, this function should be replaced by string_view.ends_with
[[nodiscard]] constexpr bool ends_with(
    const std::string_view s,
    const std::string_view suffix) noexcept {
  return (suffix.size() > s.size())
      ? false
      : suffix == s.substr(s.size() - suffix.size(), suffix.size());
}

// NOTE: In C++20, this function should be replaced by string_view.ends_with
[[nodiscard]] constexpr bool ends_with(
    const std::string_view s,
    const char prefix) noexcept {
  return !s.empty() && prefix == s.back();
}

} // namespace c10

namespace std {
template <class CharT>
struct hash<::c10::basic_string_view<CharT>> {
  size_t operator()(::c10::basic_string_view<CharT> x) const {
    return ::std::hash<::std::basic_string_view<CharT>>{}(x);
  }
};
} // namespace std
