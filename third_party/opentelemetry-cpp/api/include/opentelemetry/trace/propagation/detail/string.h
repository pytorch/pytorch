// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{
namespace propagation
{
// NOTE - code within `detail` namespace implements internal details, and not part
// of the public interface.
namespace detail
{

/**
 * Splits a string by separator, up to given buffer count words.
 * Returns the amount of words the input was split into.
 */
inline size_t SplitString(nostd::string_view s,
                          char separator,
                          nostd::string_view *results,
                          size_t count)
{
  if (count == 0)
  {
    return count;
  }

  size_t filled      = 0;
  size_t token_start = 0;
  for (size_t i = 0; i < s.size(); i++)
  {
    if (s[i] != separator)
    {
      continue;
    }

    results[filled++] = s.substr(token_start, i - token_start);

    if (filled == count)
    {
      return count;
    }

    token_start = i + 1;
  }

  if (filled < count)
  {
    results[filled++] = s.substr(token_start);
  }

  return filled;
}

}  // namespace detail
}  // namespace propagation
}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
