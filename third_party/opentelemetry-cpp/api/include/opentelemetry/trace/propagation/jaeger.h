// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef OPENTELEMETRY_NO_DEPRECATED_CODE
#  error "header <opentelemetry/trace/propagation/jaeger.h> is deprecated."
#endif

#include "detail/hex.h"
#include "detail/string.h"
#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/trace/context.h"
#include "opentelemetry/trace/default_span.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{
namespace propagation
{

static const nostd::string_view kJaegerTraceHeader = "uber-trace-id";

class OPENTELEMETRY_DEPRECATED JaegerPropagator : public context::propagation::TextMapPropagator
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

    const size_t trace_id_length = 32;
    const size_t span_id_length  = 16;

    // trace-id(32):span-id(16):0:debug(2)
    char trace_identity[trace_id_length + span_id_length + 6];
    span_context.trace_id().ToLowerBase16(
        nostd::span<char, 2 * TraceId::kSize>{&trace_identity[0], trace_id_length});
    trace_identity[trace_id_length] = ':';
    span_context.span_id().ToLowerBase16(
        nostd::span<char, 2 * SpanId::kSize>{&trace_identity[trace_id_length + 1], span_id_length});
    trace_identity[trace_id_length + span_id_length + 1] = ':';
    trace_identity[trace_id_length + span_id_length + 2] = '0';
    trace_identity[trace_id_length + span_id_length + 3] = ':';
    trace_identity[trace_id_length + span_id_length + 4] = '0';
    trace_identity[trace_id_length + span_id_length + 5] = span_context.IsSampled() ? '1' : '0';

    carrier.Set(kJaegerTraceHeader, nostd::string_view(trace_identity, sizeof(trace_identity)));
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

  bool Fields(nostd::function_ref<bool(nostd::string_view)> callback) const noexcept override
  {
    return callback(kJaegerTraceHeader);
  }

private:
  static constexpr uint8_t kIsSampled = 0x01;

  static TraceFlags GetTraceFlags(uint8_t jaeger_flags)
  {
    uint8_t sampled = jaeger_flags & kIsSampled;
    return TraceFlags(sampled);
  }

  static SpanContext ExtractImpl(const context::propagation::TextMapCarrier &carrier)
  {
    nostd::string_view trace_identity = carrier.Get(kJaegerTraceHeader);

    const size_t trace_field_count = 4;
    nostd::string_view trace_fields[trace_field_count];

    if (detail::SplitString(trace_identity, ':', trace_fields, trace_field_count) !=
        trace_field_count)
    {
      return SpanContext::GetInvalid();
    }

    nostd::string_view trace_id_hex = trace_fields[0];
    nostd::string_view span_id_hex  = trace_fields[1];
    nostd::string_view flags_hex    = trace_fields[3];

    if (!detail::IsValidHex(trace_id_hex) || !detail::IsValidHex(span_id_hex) ||
        !detail::IsValidHex(flags_hex))
    {
      return SpanContext::GetInvalid();
    }

    uint8_t trace_id[16];
    if (!detail::HexToBinary(trace_id_hex, trace_id, sizeof(trace_id)))
    {
      return SpanContext::GetInvalid();
    }

    uint8_t span_id[8];
    if (!detail::HexToBinary(span_id_hex, span_id, sizeof(span_id)))
    {
      return SpanContext::GetInvalid();
    }

    uint8_t flags;
    if (!detail::HexToBinary(flags_hex, &flags, sizeof(flags)))
    {
      return SpanContext::GetInvalid();
    }

    return SpanContext(TraceId(trace_id), SpanId(span_id), GetTraceFlags(flags), true);
  }
};

}  // namespace propagation
}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
