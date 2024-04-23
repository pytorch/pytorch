// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if __EXCEPTIONS
#  include <new>
#endif  // __EXCEPTIONS

#include <string>

#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace plugin
{
namespace detail
{
inline void CopyErrorMessage(const char *source, std::string &destination) noexcept
#if __EXCEPTIONS
try
#endif
{
  if (source == nullptr)
  {
    return;
  }
  destination.assign(source);
}
#if __EXCEPTIONS
catch (const std::bad_alloc &)
{}
#endif
}  // namespace detail
}  // namespace plugin
OPENTELEMETRY_END_NAMESPACE
