// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "opentelemetry/nostd/span.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/nostd/variant.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace common
{
/// OpenTelemetry signals can be enriched by adding attributes. The
/// \c AttributeValue type is defined as a variant of all attribute value
/// types the OpenTelemetry C++ API supports.
///
/// The following attribute value types are supported by the OpenTelemetry
/// specification:
///  - Primitive types: string, boolean, double precision floating point
///    (IEEE 754-1985) or signed 64 bit integer.
///  - Homogenous arrays of primitive type values.
///
/// \warning
/// \parblock The OpenTelemetry C++ API currently supports several attribute
/// value types that are not covered by the OpenTelemetry specification:
///  - \c uint64_t
///  - \c nostd::span<const uint64_t>
///  - \c nostd::span<uint8_t>
///
/// Those types are reserved for future use and currently should not be
/// used. There are no guarantees around how those values are handled by
/// exporters.
/// \endparblock
using AttributeValue =
    nostd::variant<bool,
                   int32_t,
                   int64_t,
                   uint32_t,
                   double,
                   const char *,
                   nostd::string_view,
                   nostd::span<const bool>,
                   nostd::span<const int32_t>,
                   nostd::span<const int64_t>,
                   nostd::span<const uint32_t>,
                   nostd::span<const double>,
                   nostd::span<const nostd::string_view>,
                   // Not currently supported by the specification, but reserved for future use.
                   // Added to provide support for all primitive C++ types.
                   uint64_t,
                   // Not currently supported by the specification, but reserved for future use.
                   // Added to provide support for all primitive C++ types.
                   nostd::span<const uint64_t>,
                   // Not currently supported by the specification, but reserved for future use.
                   // See https://github.com/open-telemetry/opentelemetry-specification/issues/780
                   nostd::span<const uint8_t>>;

enum AttributeType
{
  kTypeBool,
  kTypeInt,
  kTypeInt64,
  kTypeUInt,
  kTypeDouble,
  kTypeCString,
  kTypeString,
  kTypeSpanBool,
  kTypeSpanInt,
  kTypeSpanInt64,
  kTypeSpanUInt,
  kTypeSpanDouble,
  kTypeSpanString,
  kTypeUInt64,
  kTypeSpanUInt64,
  kTypeSpanByte
};

}  // namespace common
OPENTELEMETRY_END_NAMESPACE
