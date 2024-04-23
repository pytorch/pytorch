// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "opentelemetry/nostd/detail/type_pack_element.h"
#include "opentelemetry/nostd/detail/variant_fwd.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
template <std::size_t I, typename T>
struct variant_alternative;

template <std::size_t I, typename T>
using variant_alternative_t = typename variant_alternative<I, T>::type;

template <std::size_t I, typename T>
struct variant_alternative<I, const T> : std::add_const<variant_alternative_t<I, T>>
{};

template <std::size_t I, typename T>
struct variant_alternative<I, volatile T> : std::add_volatile<variant_alternative_t<I, T>>
{};

template <std::size_t I, typename T>
struct variant_alternative<I, const volatile T> : std::add_cv<variant_alternative_t<I, T>>
{};

template <std::size_t I, typename... Ts>
struct variant_alternative<I, variant<Ts...>>
{
  static_assert(I < sizeof...(Ts), "index out of bounds in `std::variant_alternative<>`");
  using type = detail::type_pack_element_t<I, Ts...>;
};
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
