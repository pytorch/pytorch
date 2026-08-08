#pragma once

// Deprecated. c10::ssize is retained as a thin alias for std::ssize (C++20)
// so out-of-tree code keeps building. In-tree code was migrated to std::ssize
// in #184775; new code should call std::ssize directly.

#include <iterator>

namespace c10 {

template <typename C>
[[deprecated("use std::ssize instead")]]
constexpr auto ssize(const C& c) -> decltype(std::ssize(c)) {
  return std::ssize(c);
}

} // namespace c10
