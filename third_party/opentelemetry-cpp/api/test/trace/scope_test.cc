// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/trace/scope.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/trace/noop.h"

#include <gtest/gtest.h>

using opentelemetry::trace::kSpanKey;
using opentelemetry::trace::NoopSpan;
using opentelemetry::trace::Scope;
using opentelemetry::trace::Span;
namespace nostd   = opentelemetry::nostd;
namespace context = opentelemetry::context;

TEST(ScopeTest, Construct)
{
  nostd::shared_ptr<Span> span(new NoopSpan(nullptr));
  Scope scope(span);

  context::ContextValue active_span_value = context::RuntimeContext::GetValue(kSpanKey);
  ASSERT_TRUE(nostd::holds_alternative<nostd::shared_ptr<Span>>(active_span_value));

  auto active_span = nostd::get<nostd::shared_ptr<Span>>(active_span_value);
  ASSERT_EQ(active_span, span);
}

TEST(ScopeTest, Destruct)
{
  nostd::shared_ptr<Span> span(new NoopSpan(nullptr));
  Scope scope(span);

  {
    nostd::shared_ptr<Span> span_nested(new NoopSpan(nullptr));
    Scope scope_nested(span_nested);

    context::ContextValue active_span_value = context::RuntimeContext::GetValue(kSpanKey);
    ASSERT_TRUE(nostd::holds_alternative<nostd::shared_ptr<Span>>(active_span_value));

    auto active_span = nostd::get<nostd::shared_ptr<Span>>(active_span_value);
    ASSERT_EQ(active_span, span_nested);
  }

  context::ContextValue active_span_value = context::RuntimeContext::GetValue(kSpanKey);
  ASSERT_TRUE(nostd::holds_alternative<nostd::shared_ptr<Span>>(active_span_value));

  auto active_span = nostd::get<nostd::shared_ptr<Span>>(active_span_value);
  ASSERT_EQ(active_span, span);
}
