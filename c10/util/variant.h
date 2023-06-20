#pragma once

#include <c10/util/in_place.h>

#include <cstddef>
#include <utility>
#include <variant>

namespace c10 {

template <typename... Ts>
using variant = std::variant<Ts...>;

template <std::size_t I>
constexpr std::in_place_index_t<I> in_place_index{};

template <typename T>
constexpr std::in_place_type_t<T> in_place_type{};

template <class Visitor, class... Variants>
constexpr auto visit( Visitor&& vis, Variants&&... vars ) {
    return std::visit(vis, vars...);
}

} //namespace c10
