// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <type_traits>

#include "opentelemetry/nostd/utility.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
namespace detail
{
template <std::size_t N>
using size_constant = std::integral_constant<std::size_t, N>;

template <std::size_t I, typename T>
struct indexed_type : size_constant<I>
{
  using type = T;
};

template <std::size_t I, typename... Ts>
struct type_pack_element_impl
{
private:
  template <typename>
  struct set;

  template <std::size_t... Is>
  struct set<index_sequence<Is...>> : indexed_type<Is, Ts>...
  {};

  template <typename T>
  inline static std::enable_if<true, T> impl(indexed_type<I, T>);

  inline static std::enable_if<false> impl(...);

public:
  using type = decltype(impl(set<index_sequence_for<Ts...>>{}));
};

template <std::size_t I, typename... Ts>
using type_pack_element = typename type_pack_element_impl<I, Ts...>::type;

template <std::size_t I, typename... Ts>
using type_pack_element_t = typename type_pack_element<I, Ts...>::type;
}  // namespace detail
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
