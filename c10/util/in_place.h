#pragma once

#include <utility>

namespace c10 {

using in_place_t = std::in_place_t;

template<std::size_t I>
using in_place_index_t = std::in_place_index_t<I>;

template <typename T>
using in_place_type_t = std::in_place_type_t<T>;

constexpr in_place_t in_place{};

} // namespace c10
