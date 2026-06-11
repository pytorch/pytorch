#pragma once

#include <torch/headeronly/macros/Macros.h>

#include <array>
#include <utility>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

// This helper function creates a constexpr std::array
// From a compile time list of values, without requiring you to explicitly
// write out the length.
//
// See also https://stackoverflow.com/a/26351760/23845
template <typename V, typename... T>
inline constexpr auto array_of(T&&... t) -> std::array<V, sizeof...(T)> {
  return {{std::forward<T>(t)...}};
}

HIDDEN_NAMESPACE_END(torch, headeronly)

namespace c10 {
using torch::headeronly::array_of;
} // namespace c10
