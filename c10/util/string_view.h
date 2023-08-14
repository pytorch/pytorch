#pragma once

#include <string_view>

namespace c10 {

// NOTE: This can be removed once we move to C++20 and can use std::string_view::starts_with
// in the meantime this allows us to avoid making a custom class
constexpr bool starts_with(const std::string_view& str, const std::string_view& prefix) {
  return (prefix.size() > str.size()) ? false : str.substr(0, prefix.size()) == prefix;
}

// NOTE: This can be removed once we move to C++20 and can use std::string_view::ends_with
// in the meantime this allows us to avoid making a custom class
constexpr bool ends_with(const std::string_view& str, const std::string_view& suffix) {
  return (suffix.size() > str.size()) ? false : str.substr(str.size() - suffix.size()) == suffix;
}

// NOTE: This can be removed once we move to C++20 and can use std::string_view::starts_with
// in the meantime this allows us to avoid making a custom class
constexpr bool starts_with(const std::string_view& str, const char x) {
  return !str.empty() && str.front() == x;
}

// NOTE: This can be removed once we move to C++20 and can use std::string_view::ends_with
// in the meantime this allows us to avoid making a custom class
constexpr bool ends_with(const std::string_view& str, const char x) {
  return !str.empty() && str.back() == x;
}

using string_view = std::string_view;

} // namespace c10
