// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "opentelemetry/version.h"

#define OPENTELEMETRY_RETURN(...) \
  noexcept(noexcept(__VA_ARGS__))->decltype(__VA_ARGS__) { return __VA_ARGS__; }

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
namespace detail
{
struct equal_to
{
  template <typename Lhs, typename Rhs>
  inline constexpr auto operator()(Lhs &&lhs, Rhs &&rhs) const
      OPENTELEMETRY_RETURN(std::forward<Lhs>(lhs) == std::forward<Rhs>(rhs))
};

struct not_equal_to
{
  template <typename Lhs, typename Rhs>
  inline constexpr auto operator()(Lhs &&lhs, Rhs &&rhs) const
      OPENTELEMETRY_RETURN(std::forward<Lhs>(lhs) != std::forward<Rhs>(rhs))
};

struct less
{
  template <typename Lhs, typename Rhs>
  inline constexpr auto operator()(Lhs &&lhs, Rhs &&rhs) const
      OPENTELEMETRY_RETURN(std::forward<Lhs>(lhs) < std::forward<Rhs>(rhs))
};

struct greater
{
  template <typename Lhs, typename Rhs>
  inline constexpr auto operator()(Lhs &&lhs, Rhs &&rhs) const
      OPENTELEMETRY_RETURN(std::forward<Lhs>(lhs) > std::forward<Rhs>(rhs))
};

struct less_equal
{
  template <typename Lhs, typename Rhs>
  inline constexpr auto operator()(Lhs &&lhs, Rhs &&rhs) const
      OPENTELEMETRY_RETURN(std::forward<Lhs>(lhs) <= std::forward<Rhs>(rhs))
};

struct greater_equal
{
  template <typename Lhs, typename Rhs>
  inline constexpr auto operator()(Lhs &&lhs, Rhs &&rhs) const
      OPENTELEMETRY_RETURN(std::forward<Lhs>(lhs) >= std::forward<Rhs>(rhs))
};
}  // namespace detail
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE

#undef OPENTELEMETRY_RETURN
