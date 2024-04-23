// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "opentelemetry/nostd/detail/variant_fwd.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
template <typename T>
struct variant_size;

template <typename T>
struct variant_size<const T> : variant_size<T>
{};

template <typename T>
struct variant_size<volatile T> : variant_size<T>
{};

template <typename T>
struct variant_size<const volatile T> : variant_size<T>
{};

template <typename... Ts>
struct variant_size<variant<Ts...>> : std::integral_constant<size_t, sizeof...(Ts)>
{};
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
