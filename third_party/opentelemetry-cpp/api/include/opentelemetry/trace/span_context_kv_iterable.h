// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/nostd/function_ref.h"
#include "opentelemetry/trace/span_context.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{
/**
 * Supports internal iteration over a collection of SpanContext/key-value pairs.
 */
class SpanContextKeyValueIterable
{
public:
  virtual ~SpanContextKeyValueIterable() = default;

  /**
   * Iterate over SpanContext/key-value pairs
   * @param callback a callback to invoke for each key-value for each SpanContext.
   * If the callback returns false, the iteration is aborted.
   * @return true if every SpanContext/key-value pair was iterated over
   */
  virtual bool ForEachKeyValue(
      nostd::function_ref<bool(SpanContext, const common::KeyValueIterable &)> callback)
      const noexcept = 0;
  /**
   * @return the number of key-value pairs
   */
  virtual size_t size() const noexcept = 0;
};

/**
 * @brief Null Span context that does not carry any information.
 */
class NullSpanContext : public SpanContextKeyValueIterable
{
public:
  bool ForEachKeyValue(nostd::function_ref<bool(SpanContext, const common::KeyValueIterable &)>
                       /* callback */) const noexcept override
  {
    return true;
  }

  size_t size() const noexcept override { return 0; }
};

}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
