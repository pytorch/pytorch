// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
// Standard Type aliases in nostd namespace
namespace nostd
{

// nostd::shared_ptr<T...>
template <class... _Types>
using shared_ptr = std::shared_ptr<_Types...>;

}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
