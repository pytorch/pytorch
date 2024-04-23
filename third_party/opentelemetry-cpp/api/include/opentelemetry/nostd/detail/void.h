// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
namespace detail
{
template <class...>
struct voider
{
  using type = void;
};
}  // namespace detail

/**
 * Back port of std::void_t
 *
 * Note: voider workaround is required for gcc-4.8 to make SFINAE work
 */
template <class... Tx>
using void_t = typename detail::voider<Tx...>::type;
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
