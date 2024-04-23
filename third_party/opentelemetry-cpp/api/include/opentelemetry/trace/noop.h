// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Please refer to provider.h for documentation on how to obtain a Tracer object.
//
// This file is part of the internal implementation of OpenTelemetry. Nothing in this file should be
// used directly. Please refer to span.h and tracer.h for documentation on these interfaces.

#include <memory>

#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/nostd/unique_ptr.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/span_context.h"
#include "opentelemetry/trace/span_context_kv_iterable.h"
#include "opentelemetry/trace/tracer.h"
#include "opentelemetry/trace/tracer_provider.h"
#include "opentelemetry/version.h"

namespace trace_api = opentelemetry::trace;

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{
/**
 * No-op implementation of Span. This class should not be used directly.
 */
class OPENTELEMETRY_EXPORT NoopSpan final : public Span
{
public:
  explicit NoopSpan(const std::shared_ptr<Tracer> &tracer) noexcept
      : tracer_{tracer}, span_context_{new SpanContext(false, false)}
  {}

  explicit NoopSpan(const std::shared_ptr<Tracer> &tracer,
                    nostd::unique_ptr<SpanContext> span_context) noexcept
      : tracer_{tracer}, span_context_{std::move(span_context)}
  {}

  void SetAttribute(nostd::string_view /*key*/,
                    const common::AttributeValue & /*value*/) noexcept override
  {}

  void AddEvent(nostd::string_view /*name*/) noexcept override {}

  void AddEvent(nostd::string_view /*name*/,
                common::SystemTimestamp /*timestamp*/) noexcept override
  {}

  void AddEvent(nostd::string_view /* name */,
                const common::KeyValueIterable & /* attributes */) noexcept override
  {}

  void AddEvent(nostd::string_view /*name*/,
                common::SystemTimestamp /*timestamp*/,
                const common::KeyValueIterable & /*attributes*/) noexcept override
  {}

#if OPENTELEMETRY_ABI_VERSION_NO >= 2
  void AddLink(const SpanContext & /* target */,
               const common::KeyValueIterable & /* attrs */) noexcept override
  {}

  void AddLinks(const SpanContextKeyValueIterable & /* links */) noexcept override {}
#endif

  void SetStatus(StatusCode /*code*/, nostd::string_view /*description*/) noexcept override {}

  void UpdateName(nostd::string_view /*name*/) noexcept override {}

  void End(const EndSpanOptions & /*options*/) noexcept override {}

  bool IsRecording() const noexcept override { return false; }

  SpanContext GetContext() const noexcept override { return *span_context_.get(); }

private:
  std::shared_ptr<Tracer> tracer_;
  nostd::unique_ptr<SpanContext> span_context_;
};

/**
 * No-op implementation of Tracer.
 */
class OPENTELEMETRY_EXPORT NoopTracer final : public Tracer,
                                              public std::enable_shared_from_this<NoopTracer>
{
public:
  // Tracer
  nostd::shared_ptr<Span> StartSpan(nostd::string_view /*name*/,
                                    const common::KeyValueIterable & /*attributes*/,
                                    const SpanContextKeyValueIterable & /*links*/,
                                    const StartSpanOptions & /*options*/) noexcept override
  {
    // Don't allocate a no-op span for every StartSpan call, but use a static
    // singleton for this case.
    static nostd::shared_ptr<trace::Span> noop_span(new trace::NoopSpan{this->shared_from_this()});

    return noop_span;
  }

  void ForceFlushWithMicroseconds(uint64_t /*timeout*/) noexcept override {}

  void CloseWithMicroseconds(uint64_t /*timeout*/) noexcept override {}
};

/**
 * No-op implementation of a TracerProvider.
 */
class OPENTELEMETRY_EXPORT NoopTracerProvider final : public trace::TracerProvider
{
public:
  NoopTracerProvider() noexcept
      : tracer_{nostd::shared_ptr<trace::NoopTracer>(new trace::NoopTracer)}
  {}

#if OPENTELEMETRY_ABI_VERSION_NO >= 2
  nostd::shared_ptr<trace::Tracer> GetTracer(
      nostd::string_view /* name */,
      nostd::string_view /* version */,
      nostd::string_view /* schema_url */,
      const common::KeyValueIterable * /* attributes */) noexcept override
  {
    return tracer_;
  }
#else
  nostd::shared_ptr<trace::Tracer> GetTracer(nostd::string_view /* name */,
                                             nostd::string_view /* version */,
                                             nostd::string_view /* schema_url */) noexcept override
  {
    return tracer_;
  }
#endif

private:
  nostd::shared_ptr<trace::Tracer> tracer_;
};
}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
