// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>

#include "opentelemetry/context/context.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/trace/default_span.h"
#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/span_context_kv_iterable_view.h"
#include "opentelemetry/trace/span_startoptions.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{
/**
 * Handles span creation and in-process context propagation.
 *
 * This class provides methods for manipulating the context, creating spans, and controlling spans'
 * lifecycles.
 */
class Tracer
{
public:
  virtual ~Tracer() = default;
  /**
   * Starts a span.
   *
   * Optionally sets attributes at Span creation from the given key/value pairs.
   *
   * Attributes will be processed in order, previous attributes with the same
   * key will be overwritten.
   */
  virtual nostd::shared_ptr<Span> StartSpan(nostd::string_view name,
                                            const common::KeyValueIterable &attributes,
                                            const SpanContextKeyValueIterable &links,
                                            const StartSpanOptions &options = {}) noexcept = 0;

  nostd::shared_ptr<Span> StartSpan(nostd::string_view name,
                                    const StartSpanOptions &options = {}) noexcept
  {
    return this->StartSpan(name, {}, {}, options);
  }

  template <class T,
            nostd::enable_if_t<common::detail::is_key_value_iterable<T>::value> * = nullptr>
  nostd::shared_ptr<Span> StartSpan(nostd::string_view name,
                                    const T &attributes,
                                    const StartSpanOptions &options = {}) noexcept
  {
    return this->StartSpan(name, attributes, {}, options);
  }

  nostd::shared_ptr<Span> StartSpan(nostd::string_view name,
                                    const common::KeyValueIterable &attributes,
                                    const StartSpanOptions &options = {}) noexcept
  {
    return this->StartSpan(name, attributes, NullSpanContext(), options);
  }

  template <class T,
            class U,
            nostd::enable_if_t<common::detail::is_key_value_iterable<T>::value> * = nullptr,
            nostd::enable_if_t<detail::is_span_context_kv_iterable<U>::value> *   = nullptr>
  nostd::shared_ptr<Span> StartSpan(nostd::string_view name,
                                    const T &attributes,
                                    const U &links,
                                    const StartSpanOptions &options = {}) noexcept
  {
    return this->StartSpan(name, common::KeyValueIterableView<T>(attributes),
                           SpanContextKeyValueIterableView<U>(links), options);
  }

  nostd::shared_ptr<Span> StartSpan(
      nostd::string_view name,
      std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>> attributes,
      const StartSpanOptions &options = {}) noexcept
  {

    return this->StartSpan(name, attributes, {}, options);
  }

  template <class T,
            nostd::enable_if_t<common::detail::is_key_value_iterable<T>::value> * = nullptr>
  nostd::shared_ptr<Span> StartSpan(
      nostd::string_view name,
      const T &attributes,
      std::initializer_list<
          std::pair<SpanContext,
                    std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>>>>
          links,
      const StartSpanOptions &options = {}) noexcept
  {
    return this->StartSpan(
        name, attributes,
        nostd::span<const std::pair<SpanContext, std::initializer_list<std::pair<
                                                     nostd::string_view, common::AttributeValue>>>>{
            links.begin(), links.end()},
        options);
  }

  template <class T,
            nostd::enable_if_t<common::detail::is_key_value_iterable<T>::value> * = nullptr>
  nostd::shared_ptr<Span> StartSpan(
      nostd::string_view name,
      std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>> attributes,
      const T &links,
      const StartSpanOptions &options = {}) noexcept
  {
    return this->StartSpan(name,
                           nostd::span<const std::pair<nostd::string_view, common::AttributeValue>>{
                               attributes.begin(), attributes.end()},
                           links, options);
  }

  nostd::shared_ptr<Span> StartSpan(
      nostd::string_view name,
      std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>> attributes,
      std::initializer_list<
          std::pair<SpanContext,
                    std::initializer_list<std::pair<nostd::string_view, common::AttributeValue>>>>
          links,
      const StartSpanOptions &options = {}) noexcept
  {
    return this->StartSpan(
        name,
        nostd::span<const std::pair<nostd::string_view, common::AttributeValue>>{attributes.begin(),
                                                                                 attributes.end()},
        nostd::span<const std::pair<SpanContext, std::initializer_list<std::pair<
                                                     nostd::string_view, common::AttributeValue>>>>{
            links.begin(), links.end()},
        options);
  }

  /**
   * Set the active span. The span will remain active until the returned Scope
   * object is destroyed.
   * @param span the span that should be set as the new active span.
   * @return a Scope that controls how long the span will be active.
   */
  static Scope WithActiveSpan(nostd::shared_ptr<Span> &span) noexcept { return Scope{span}; }

  /**
   * Get the currently active span.
   * @return the currently active span, or an invalid default span if no span
   * is active.
   */
  static nostd::shared_ptr<Span> GetCurrentSpan() noexcept
  {
    context::ContextValue active_span = context::RuntimeContext::GetValue(kSpanKey);
    if (nostd::holds_alternative<nostd::shared_ptr<Span>>(active_span))
    {
      return nostd::get<nostd::shared_ptr<Span>>(active_span);
    }
    else
    {
      return nostd::shared_ptr<Span>(new DefaultSpan(SpanContext::GetInvalid()));
    }
  }

  /**
   * Force any buffered spans to flush.
   * @param timeout to complete the flush
   */
  template <class Rep, class Period>
  void ForceFlush(std::chrono::duration<Rep, Period> timeout) noexcept
  {
    this->ForceFlushWithMicroseconds(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(timeout).count()));
  }

  virtual void ForceFlushWithMicroseconds(uint64_t timeout) noexcept = 0;

  /**
   * ForceFlush any buffered spans and stop reporting spans.
   * @param timeout to complete the flush
   */
  template <class Rep, class Period>
  void Close(std::chrono::duration<Rep, Period> timeout) noexcept
  {
    this->CloseWithMicroseconds(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(timeout).count()));
  }

  virtual void CloseWithMicroseconds(uint64_t timeout) noexcept = 0;
};
}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
