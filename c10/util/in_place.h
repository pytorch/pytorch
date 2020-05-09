#pragma once

#include <cstddef>

namespace c10 {

struct in_place_t { explicit in_place_t() = default; };

template <std::size_t I>
struct in_place_index_t { explicit in_place_index_t() = default; };

template <typename T>
struct in_place_type_t { explicit in_place_type_t() = default; };

constexpr in_place_t in_place{};

} // namespace c10
