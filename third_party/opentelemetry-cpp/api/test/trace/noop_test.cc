// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/trace/noop.h"
#include "opentelemetry/common/timestamp.h"

#include <map>
#include <memory>
#include <string>

#include <gtest/gtest.h>

namespace trace_api = opentelemetry::trace;
namespace nonstd    = opentelemetry::nostd;
namespace common    = opentelemetry::common;

TEST(NoopTest, UseNoopTracers)
{
  std::shared_ptr<trace_api::Tracer> tracer{new trace_api::NoopTracer{}};
  auto s1 = tracer->StartSpan("abc");

  std::map<std::string, std::string> attributes1;
  s1->AddEvent("abc", attributes1);

  std::vector<std::pair<std::string, int>> attributes2;
  s1->AddEvent("abc", attributes2);

  s1->AddEvent("abc", {{"a", 1}, {"b", "2"}, {"c", 3.0}});

  std::vector<std::pair<std::string, std::vector<int>>> attributes3;
  s1->AddEvent("abc", attributes3);

  s1->SetAttribute("abc", 4);

  s1->AddEvent("abc");  // add Empty

  EXPECT_EQ(s1->IsRecording(), false);

  s1->SetStatus(trace_api::StatusCode::kUnset, "span unset");

  s1->UpdateName("test_name");

  common::SystemTimestamp t1;
  s1->AddEvent("test_time_stamp", t1);

  s1->GetContext();
}

#if OPENTELEMETRY_ABI_VERSION_NO >= 2
TEST(NoopTest, UseNoopTracersAbiv2)
{
  std::shared_ptr<trace_api::Tracer> tracer{new trace_api::NoopTracer{}};
  auto s1 = tracer->StartSpan("abc");

  EXPECT_EQ(s1->IsRecording(), false);

  trace_api::SpanContext target(false, false);
  s1->AddLink(target, {{"noop1", 1}});

  s1->AddLinks({{trace_api::SpanContext(false, false), {{"noop2", 2}}}});
}
#endif /* OPENTELEMETRY_ABI_VERSION_NO >= 2 */

TEST(NoopTest, StartSpan)
{
  std::shared_ptr<trace_api::Tracer> tracer{new trace_api::NoopTracer{}};

  std::map<std::string, std::string> attrs = {{"a", "3"}};
  std::vector<std::pair<trace_api::SpanContext, std::map<std::string, std::string>>> links = {
      {trace_api::SpanContext(false, false), attrs}};
  auto s1 = tracer->StartSpan("abc", attrs, links);

  auto s2 =
      tracer->StartSpan("efg", {{"a", 3}}, {{trace_api::SpanContext(false, false), {{"b", 4}}}});
}

TEST(NoopTest, CreateSpanValidSpanContext)
{
  // Create valid spancontext for NoopSpan

  constexpr uint8_t buf_span[]  = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr uint8_t buf_trace[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  auto trace_id                 = trace_api::TraceId{buf_trace};
  auto span_id                  = trace_api::SpanId{buf_span};
  auto span_context             = nonstd::unique_ptr<trace_api::SpanContext>(
      new trace_api::SpanContext{trace_id, span_id, trace_api::TraceFlags{true}, false});
  std::shared_ptr<trace_api::Tracer> tracer{new trace_api::NoopTracer{}};
  auto s1 =
      nonstd::shared_ptr<trace_api::Span>(new trace_api::NoopSpan(tracer, std::move(span_context)));
  auto stored_span_context = s1->GetContext();
  EXPECT_EQ(stored_span_context.span_id(), span_id);
  EXPECT_EQ(stored_span_context.trace_id(), trace_id);

  s1->AddEvent("even1");  // noop
  s1->End();              // noop
}
