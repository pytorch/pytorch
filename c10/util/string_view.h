#pragma once

#include <functional>
#include <string_view>

namespace c10 {

struct TransparentStringHash {
  using is_transparent = void;

  size_t operator()(std::string_view value) const noexcept {
    return std::hash<std::string_view>{}(value);
  }
};

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
