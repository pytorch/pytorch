#pragma once

#include <array>
#include <utility>

namespace c10 {
namespace guts {

template <typename T, int N>
using array = std::array<T, N>;

} // namespace guts
} // namespace c10
