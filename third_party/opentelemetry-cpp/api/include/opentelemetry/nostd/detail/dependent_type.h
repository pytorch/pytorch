// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
namespace detail
{
template <typename T, bool>
struct dependent_type : T
{};
}  // namespace detail
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
