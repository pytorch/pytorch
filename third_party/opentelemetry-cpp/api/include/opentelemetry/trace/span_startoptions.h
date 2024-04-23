// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/common/timestamp.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/nostd/variant.h"
#include "opentelemetry/trace/span_context.h"
#include "opentelemetry/trace/span_metadata.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{

/**
 * StartSpanOptions provides options to set properties of a Span at the time of
 * its creation
 */
struct StartSpanOptions
{
  // Optionally sets the start time of a Span.
  //
  // If the start time of a Span is set, timestamps from both the system clock
  // and steady clock must be provided.
  //
  // Timestamps from the steady clock can be used to most accurately measure a
  // Span's duration, while timestamps from the system clock can be used to most
  // accurately place a Span's
  // time point relative to other Spans collected across a distributed system.
  common::SystemTimestamp start_system_time;
  common::SteadyTimestamp start_steady_time;

  // Explicitly set the parent of a Span.
  //
  // The `parent` field is designed to establish  parent-child relationships
  // in tracing spans. It can be set to either a `SpanContext` or a
  // `context::Context` object.
  //
  // - When set to valid `SpanContext`, it directly assigns a specific Span as the parent
  // of the newly created Span.
  //
  // - Alternatively, setting the `parent` field to a `context::Context` allows for
  // more nuanced parent identification:
  //   1. If the `Context` contains a Span object, this Span is treated as the parent.
  //   2. If the `Context` contains the boolean flag `is_root_span` set to `true`,
  //      it indicates that the new Span should be treated as a root Span, i.e., it
  //      does not have a parent Span.
  //   Example Usage:
  //   ```cpp
  //   trace_api::StartSpanOptions options;
  //   opentelemetry::context::Context root;
  //   root                    = root.SetValue(kIsRootSpanKey, true);
  //   options.parent = root;
  //   auto root_span = tracer->StartSpan("span root", options);
  //  ```
  //
  // - If the `parent` field is not set, the newly created Span will inherit the
  // parent of the currently active Span (if any) in the current context.
  //
  nostd::variant<SpanContext, context::Context> parent = SpanContext::GetInvalid();

  // TODO:
  // SpanContext remote_parent;
  // Links
  SpanKind kind = SpanKind::kInternal;
};

}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
