// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "opentelemetry/nostd/type_traits.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
namespace detail
{
enum class Trait
{
  TriviallyAvailable,
  Available,
  Unavailable
};

template <typename T,
          template <typename>
          class IsTriviallyAvailable,
          template <typename>
          class IsAvailable>
inline constexpr Trait trait()
{
  return IsTriviallyAvailable<T>::value
             ? Trait::TriviallyAvailable
             : IsAvailable<T>::value ? Trait::Available : Trait::Unavailable;
}

inline constexpr Trait common_trait_impl(Trait result)
{
  return result;
}

template <typename... Traits>
inline constexpr Trait common_trait_impl(Trait result, Trait t, Traits... ts)
{
  return static_cast<int>(t) > static_cast<int>(result) ? common_trait_impl(t, ts...)
                                                        : common_trait_impl(result, ts...);
}

template <typename... Traits>
inline constexpr Trait common_trait(Traits... ts)
{
  return common_trait_impl(Trait::TriviallyAvailable, ts...);
}

template <typename... Ts>
struct traits
{
  static constexpr Trait copy_constructible_trait =
      common_trait(trait<Ts, is_trivially_copy_constructible, std::is_copy_constructible>()...);

  static constexpr Trait move_constructible_trait =
      common_trait(trait<Ts, is_trivially_move_constructible, std::is_move_constructible>()...);

  static constexpr Trait copy_assignable_trait =
      common_trait(copy_constructible_trait,
                   trait<Ts, is_trivially_copy_assignable, std::is_copy_assignable>()...);

  static constexpr Trait move_assignable_trait =
      common_trait(move_constructible_trait,
                   trait<Ts, is_trivially_move_assignable, std::is_move_assignable>()...);

  static constexpr Trait destructible_trait =
      common_trait(trait<Ts, std::is_trivially_destructible, std::is_destructible>()...);
};
}  // namespace detail
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
