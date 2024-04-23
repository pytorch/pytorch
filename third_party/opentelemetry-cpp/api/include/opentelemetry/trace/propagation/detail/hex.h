// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>

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

constexpr int8_t kHexDigits[256] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  -1, -1, -1, -1, -1, -1, -1, 10, 11, 12, 13, 14, 15, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};

inline int8_t HexToInt(char c)
{
  return kHexDigits[uint8_t(c)];
}

inline bool IsValidHex(nostd::string_view s)
{
  return std::all_of(s.begin(), s.end(), [](char c) { return HexToInt(c) != -1; });
}

/**
 * Converts a hexadecimal to binary format if the hex string will fit the buffer.
 * Smaller hex strings are left padded with zeroes.
 */
inline bool HexToBinary(nostd::string_view hex, uint8_t *buffer, size_t buffer_size)
{
  std::memset(buffer, 0, buffer_size);

  if (hex.size() > buffer_size * 2)
  {
    return false;
  }

  int64_t hex_size     = int64_t(hex.size());
  int64_t buffer_pos   = int64_t(buffer_size) - (hex_size + 1) / 2;
  int64_t last_hex_pos = hex_size - 1;

  bool is_hex_size_odd = (hex_size % 2) == 1;
  int64_t i            = 0;

  if (is_hex_size_odd)
  {
    buffer[buffer_pos++] = HexToInt(hex[i++]);
  }

  for (; i < last_hex_pos; i += 2)
  {
    buffer[buffer_pos++] = (HexToInt(hex[i]) << 4) | HexToInt(hex[i + 1]);
  }

  return true;
}

}  // namespace detail
}  // namespace propagation
}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
