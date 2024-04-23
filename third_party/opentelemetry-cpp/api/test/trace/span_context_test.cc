// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/trace/span_context.h"
#include "opentelemetry/trace/span_id.h"
#include "opentelemetry/trace/trace_id.h"

#include <gtest/gtest.h>

using opentelemetry::trace::SpanContext;
namespace trace_api = opentelemetry::trace;

TEST(SpanContextTest, IsSampled)
{
  SpanContext s1(true, true);

  ASSERT_EQ(s1.IsSampled(), true);

  SpanContext s2(false, true);

  ASSERT_EQ(s2.IsSampled(), false);
}

TEST(SpanContextTest, IsRemote)
{
  SpanContext s1(true, true);

  ASSERT_EQ(s1.IsRemote(), true);

  SpanContext s2(true, false);

  ASSERT_EQ(s2.IsRemote(), false);
}

TEST(SpanContextTest, TraceFlags)
{
  SpanContext s1(true, true);

  ASSERT_EQ(s1.trace_flags().flags(), 1);

  SpanContext s2(false, true);

  ASSERT_EQ(s2.trace_flags().flags(), 0);
}

// Test that SpanContext is invalid
TEST(SpanContextTest, Invalid)
{
  SpanContext s1 = SpanContext::GetInvalid();
  EXPECT_FALSE(s1.IsValid());

  // Test that trace id and span id are invalid
  EXPECT_EQ(s1.trace_id(), trace_api::TraceId());
  EXPECT_EQ(s1.span_id(), trace_api::SpanId());
}
