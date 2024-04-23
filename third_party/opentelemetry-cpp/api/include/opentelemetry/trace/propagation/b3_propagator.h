// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "detail/hex.h"
#include "detail/string.h"
#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/trace/context.h"
#include "opentelemetry/trace/default_span.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{
namespace propagation
{

static const nostd::string_view kB3CombinedHeader = "b3";

static const nostd::string_view kB3TraceIdHeader = "X-B3-TraceId";
static const nostd::string_view kB3SpanIdHeader  = "X-B3-SpanId";
static const nostd::string_view kB3SampledHeader = "X-B3-Sampled";

/*
     B3, single header:
                   b3: 80f198ee56343ba864fe8b2a57d3eff7-e457b5a2e4d86bd1-1-05e3ac9a4f6e3b90
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^ ^ ^^^^^^^^^^^^^^^^
                       0          TraceId            31 33  SpanId    48 | 52 ParentSpanId 68
                                                                        50 Debug flag
     Multiheader version:                                           X-B3-Sampled
                             X-B3-TraceId                X-B3-SpanId    X-B3-ParentSpanId (ignored)
*/

static const int kTraceIdHexStrLength = 32;
static const int kSpanIdHexStrLength  = 16;

// The B3PropagatorExtractor class provides an interface that enables extracting context from
// headers of HTTP requests. HTTP frameworks and clients can integrate with B3Propagator by
// providing the object containing the headers, and a getter function for the extraction. Based on:
// https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/context/api-propagators.md#b3-extract

class B3PropagatorExtractor : public context::propagation::TextMapPropagator
{
public:
  // Returns the context that is stored in the HTTP header carrier.
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
    uint8_t buf[kTraceIdHexStrLength / 2];
    detail::HexToBinary(trace_id, buf, sizeof(buf));
    return TraceId(buf);
  }

  static SpanId SpanIdFromHex(nostd::string_view span_id)
  {
    uint8_t buf[kSpanIdHexStrLength / 2];
    detail::HexToBinary(span_id, buf, sizeof(buf));
    return SpanId(buf);
  }

  static TraceFlags TraceFlagsFromHex(nostd::string_view trace_flags)
  {
    if (trace_flags.length() != 1 || (trace_flags[0] != '1' && trace_flags[0] != 'd'))
    {  // check for invalid length of flags and treat 'd' as sampled
      return TraceFlags(0);
    }
    return TraceFlags(TraceFlags::kIsSampled);
  }

private:
  static SpanContext ExtractImpl(const context::propagation::TextMapCarrier &carrier)
  {
    nostd::string_view trace_id_hex;
    nostd::string_view span_id_hex;
    nostd::string_view trace_flags_hex;

    // first let's try a single-header variant
    auto singleB3Header = carrier.Get(kB3CombinedHeader);
    if (!singleB3Header.empty())
    {
      std::array<nostd::string_view, 3> fields{};
      // https://github.com/openzipkin/b3-propagation/blob/master/RATIONALE.md
      if (detail::SplitString(singleB3Header, '-', fields.data(), 3) < 2)
      {
        return SpanContext::GetInvalid();
      }

      trace_id_hex    = fields[0];
      span_id_hex     = fields[1];
      trace_flags_hex = fields[2];
    }
    else
    {
      trace_id_hex    = carrier.Get(kB3TraceIdHeader);
      span_id_hex     = carrier.Get(kB3SpanIdHeader);
      trace_flags_hex = carrier.Get(kB3SampledHeader);
    }

    if (!detail::IsValidHex(trace_id_hex) || !detail::IsValidHex(span_id_hex))
    {
      return SpanContext::GetInvalid();
    }

    TraceId trace_id = TraceIdFromHex(trace_id_hex);
    SpanId span_id   = SpanIdFromHex(span_id_hex);

    if (!trace_id.IsValid() || !span_id.IsValid())
    {
      return SpanContext::GetInvalid();
    }

    return SpanContext(trace_id, span_id, TraceFlagsFromHex(trace_flags_hex), true);
  }
};

// The B3Propagator class provides interface that enables extracting and injecting context into
// single header of HTTP Request.
class B3Propagator : public B3PropagatorExtractor
{
public:
  // Sets the context for a HTTP header carrier with self defined rules.
  void Inject(context::propagation::TextMapCarrier &carrier,
              const context::Context &context) noexcept override
  {
    SpanContext span_context = trace::GetSpan(context)->GetContext();
    if (!span_context.IsValid())
    {
      return;
    }

    char trace_identity[kTraceIdHexStrLength + kSpanIdHexStrLength + 3];
    static_assert(sizeof(trace_identity) == 51, "b3 trace identity buffer size mismatch");
    span_context.trace_id().ToLowerBase16(nostd::span<char, 2 * TraceId::kSize>{
        &trace_identity[0], static_cast<std::size_t>(kTraceIdHexStrLength)});
    trace_identity[kTraceIdHexStrLength] = '-';
    span_context.span_id().ToLowerBase16(nostd::span<char, 2 * SpanId::kSize>{
        &trace_identity[kTraceIdHexStrLength + 1], static_cast<std::size_t>(kSpanIdHexStrLength)});
    trace_identity[kTraceIdHexStrLength + kSpanIdHexStrLength + 1] = '-';
    trace_identity[kTraceIdHexStrLength + kSpanIdHexStrLength + 2] =
        span_context.trace_flags().IsSampled() ? '1' : '0';

    carrier.Set(kB3CombinedHeader, nostd::string_view(trace_identity, sizeof(trace_identity)));
  }

  bool Fields(nostd::function_ref<bool(nostd::string_view)> callback) const noexcept override
  {
    return callback(kB3CombinedHeader);
  }
};

class B3PropagatorMultiHeader : public B3PropagatorExtractor
{
public:
  void Inject(context::propagation::TextMapCarrier &carrier,
              const context::Context &context) noexcept override
  {
    SpanContext span_context = GetSpan(context)->GetContext();
    if (!span_context.IsValid())
    {
      return;
    }
    char trace_id[32];
    TraceId(span_context.trace_id()).ToLowerBase16(trace_id);
    char span_id[16];
    SpanId(span_context.span_id()).ToLowerBase16(span_id);
    char trace_flags[2];
    TraceFlags(span_context.trace_flags()).ToLowerBase16(trace_flags);
    carrier.Set(kB3TraceIdHeader, nostd::string_view(trace_id, sizeof(trace_id)));
    carrier.Set(kB3SpanIdHeader, nostd::string_view(span_id, sizeof(span_id)));
    carrier.Set(kB3SampledHeader, nostd::string_view(trace_flags + 1, 1));
  }

  bool Fields(nostd::function_ref<bool(nostd::string_view)> callback) const noexcept override
  {
    return callback(kB3TraceIdHeader) && callback(kB3SpanIdHeader) && callback(kB3SampledHeader);
  }
};

}  // namespace propagation
}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
