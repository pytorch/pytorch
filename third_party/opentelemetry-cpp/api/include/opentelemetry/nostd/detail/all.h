// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "opentelemetry/nostd/utility.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
namespace detail
{
template <bool... Bs>
using all = std::is_same<integer_sequence<bool, true, Bs...>, integer_sequence<bool, Bs..., true>>;

}  // namespace detail
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
