// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/common/timestamp.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{

enum class SpanKind
{
  kInternal,
  kServer,
  kClient,
  kProducer,
  kConsumer,
};

// The key identifies the active span in the current context.
constexpr char kSpanKey[]       = "active_span";
constexpr char kIsRootSpanKey[] = "is_root_span";

// StatusCode - Represents the canonical set of status codes of a finished Span.
enum class StatusCode
{
  kUnset,  // default status
  kOk,     // Operation has completed successfully.
  kError   // The operation contains an error
};

/**
 * EndSpanOptions provides options to set properties of a Span when it is
 * ended.
 */
struct EndSpanOptions
{
  // Optionally sets the end time of a Span.
  common::SteadyTimestamp end_steady_time;
};

}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
