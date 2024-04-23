// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/context/context.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/trace/default_span.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{

// Get Span from explicit context
inline nostd::shared_ptr<Span> GetSpan(const context::Context &context) noexcept
{
  context::ContextValue span = context.GetValue(kSpanKey);
  if (nostd::holds_alternative<nostd::shared_ptr<Span>>(span))
  {
    return nostd::get<nostd::shared_ptr<Span>>(span);
  }
  return nostd::shared_ptr<Span>(new DefaultSpan(SpanContext::GetInvalid()));
}

inline bool IsRootSpan(const context::Context &context) noexcept
{
  context::ContextValue is_root_span = context.GetValue(kIsRootSpanKey);
  if (nostd::holds_alternative<bool>(is_root_span))
  {
    return nostd::get<bool>(is_root_span);
  }
  return false;
}

// Set Span into explicit context
inline context::Context SetSpan(context::Context &context, nostd::shared_ptr<Span> span) noexcept
{
  return context.SetValue(kSpanKey, span);
}

}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
