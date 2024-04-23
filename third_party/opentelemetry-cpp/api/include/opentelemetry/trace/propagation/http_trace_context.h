// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "detail/hex.h"
#include "detail/string.h"
#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/nostd/function_ref.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/span.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/trace/context.h"
#include "opentelemetry/trace/default_span.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{
namespace propagation
{
static const nostd::string_view kTraceParent = "traceparent";
static const nostd::string_view kTraceState  = "tracestate";
static const size_t kVersionSize             = 2;
static const size_t kTraceIdSize             = 32;
static const size_t kSpanIdSize              = 16;
static const size_t kTraceFlagsSize          = 2;
static const size_t kTraceParentSize         = 55;

// The HttpTraceContext provides methods to extract and inject
// context into headers of HTTP requests with traces.
// Example:
//    HttpTraceContext().Inject(carrier, context);
//    HttpTraceContext().Extract(carrier, context);

class HttpTraceContext : public context::propagation::TextMapPropagator
{
public:
  void Inject(context::propagation::TextMapCarrier &carrier,
              const context::Context &context) noexcept override
  {
    SpanContext span_context = trace::GetSpan(context)->GetContext();
    if (!span_context.IsValid())
    {
      return;
    }
    InjectImpl(carrier, span_context);
  }

  context::Context Extract(const context::propagation::TextMapCarrier &carrier,
                           context::Context &context) noexcept override
  {
    SpanContext span_context = ExtractImpl(carrier);
    nostd::shared_ptr<Span> sp{new DefaultSpan(span_context)};
    if (span_context.IsValid())
    {
      return trace::SetSpan(context, sp);
    }
    else
    {
      return context;
    }
  }

  static TraceId TraceIdFromHex(nostd::string_view trace_id)
  {
    uint8_t buf[kTraceIdSize / 2];
    detail::HexToBinary(trace_id, buf, sizeof(buf));
    return TraceId(buf);
  }

  static SpanId SpanIdFromHex(nostd::string_view span_id)
  {
    uint8_t buf[kSpanIdSize / 2];
    detail::HexToBinary(span_id, buf, sizeof(buf));
    return SpanId(buf);
  }

  static TraceFlags TraceFlagsFromHex(nostd::string_view trace_flags)
  {
    uint8_t flags;
    detail::HexToBinary(trace_flags, &flags, sizeof(flags));
    return TraceFlags(flags);
  }

private:
  static constexpr uint8_t kInvalidVersion = 0xFF;

  static bool IsValidVersion(nostd::string_view version_hex)
  {
    uint8_t version;
    detail::HexToBinary(version_hex, &version, sizeof(version));
    return version != kInvalidVersion;
  }

  static void InjectImpl(context::propagation::TextMapCarrier &carrier,
                         const SpanContext &span_context)
  {
    char trace_parent[kTraceParentSize];
    trace_parent[0] = '0';
    trace_parent[1] = '0';
    trace_parent[2] = '-';
    span_context.trace_id().ToLowerBase16(
        nostd::span<char, 2 * TraceId::kSize>{&trace_parent[3], kTraceIdSize});
    trace_parent[kTraceIdSize + 3] = '-';
    span_context.span_id().ToLowerBase16(
        nostd::span<char, 2 * SpanId::kSize>{&trace_parent[kTraceIdSize + 4], kSpanIdSize});
    trace_parent[kTraceIdSize + kSpanIdSize + 4] = '-';
    span_context.trace_flags().ToLowerBase16(
        nostd::span<char, 2>{&trace_parent[kTraceIdSize + kSpanIdSize + 5], 2});

    carrier.Set(kTraceParent, nostd::string_view(trace_parent, sizeof(trace_parent)));
    const auto trace_state = span_context.trace_state()->ToHeader();
    if (!trace_state.empty())
    {
      carrier.Set(kTraceState, trace_state);
    }
  }

  static SpanContext ExtractContextFromTraceHeaders(nostd::string_view trace_parent,
                                                    nostd::string_view trace_state)
  {
    if (trace_parent.size() != kTraceParentSize)
    {
      return SpanContext::GetInvalid();
    }

    std::array<nostd::string_view, 4> fields{};
    if (detail::SplitString(trace_parent, '-', fields.data(), 4) != 4)
    {
      return SpanContext::GetInvalid();
    }

    nostd::string_view version_hex     = fields[0];
    nostd::string_view trace_id_hex    = fields[1];
    nostd::string_view span_id_hex     = fields[2];
    nostd::string_view trace_flags_hex = fields[3];

    if (version_hex.size() != kVersionSize || trace_id_hex.size() != kTraceIdSize ||
        span_id_hex.size() != kSpanIdSize || trace_flags_hex.size() != kTraceFlagsSize)
    {
      return SpanContext::GetInvalid();
    }

    if (!detail::IsValidHex(version_hex) || !detail::IsValidHex(trace_id_hex) ||
        !detail::IsValidHex(span_id_hex) || !detail::IsValidHex(trace_flags_hex))
    {
      return SpanContext::GetInvalid();
    }

    if (!IsValidVersion(version_hex))
    {
      return SpanContext::GetInvalid();
    }

    TraceId trace_id = TraceIdFromHex(trace_id_hex);
    SpanId span_id   = SpanIdFromHex(span_id_hex);

    if (!trace_id.IsValid() || !span_id.IsValid())
    {
      return SpanContext::GetInvalid();
    }

    return SpanContext(trace_id, span_id, TraceFlagsFromHex(trace_flags_hex), true,
                       trace::TraceState::FromHeader(trace_state));
  }

  static SpanContext ExtractImpl(const context::propagation::TextMapCarrier &carrier)
  {
    nostd::string_view trace_parent = carrier.Get(kTraceParent);
    nostd::string_view trace_state  = carrier.Get(kTraceState);
    if (trace_parent == "")
    {
      return SpanContext::GetInvalid();
    }

    return ExtractContextFromTraceHeaders(trace_parent, trace_state);
  }

  bool Fields(nostd::function_ref<bool(nostd::string_view)> callback) const noexcept override
  {
    return (callback(kTraceParent) && callback(kTraceState));
  }
};
}  // namespace propagation
}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
