// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
// Standard Type aliases in nostd namespace
namespace nostd
{

// nostd::unique_ptr<T...>
template <class... _Types>
using unique_ptr = std::unique_ptr<_Types...>;

}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
