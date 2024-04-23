// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
template <class T>
using decay_t = typename std::decay<T>::type;
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
