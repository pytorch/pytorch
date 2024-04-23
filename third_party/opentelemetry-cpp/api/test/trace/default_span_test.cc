// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/trace/default_span.h"
#include "opentelemetry/trace/span_context.h"

#include <cstring>
#include <string>

#include <gtest/gtest.h>

namespace
{

using opentelemetry::trace::DefaultSpan;
using opentelemetry::trace::SpanContext;

TEST(DefaultSpanTest, GetContext)
{
  SpanContext span_context = SpanContext(false, false);
  DefaultSpan sp           = DefaultSpan(span_context);
  EXPECT_EQ(span_context, sp.GetContext());
}
}  // namespace
