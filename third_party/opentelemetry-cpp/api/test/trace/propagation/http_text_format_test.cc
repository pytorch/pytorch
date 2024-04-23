// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/trace/context.h"
#include "opentelemetry/trace/propagation/http_trace_context.h"
#include "opentelemetry/trace/scope.h"
#include "util.h"

#include <map>
#include <unordered_map>

#include <gtest/gtest.h>

using namespace opentelemetry;

class TextMapCarrierTest : public context::propagation::TextMapCarrier
{
public:
  virtual nostd::string_view Get(nostd::string_view key) const noexcept override
  {
    auto it = headers_.find(std::string(key));
    if (it != headers_.end())
    {
      return nostd::string_view(it->second);
    }
    return "";
  }
  virtual void Set(nostd::string_view key, nostd::string_view value) noexcept override
  {
    headers_[std::string(key)] = std::string(value);
  }

  std::map<std::string, std::string> headers_;
};

using MapHttpTraceContext = trace::propagation::HttpTraceContext;

static MapHttpTraceContext format = MapHttpTraceContext();

TEST(TextMapPropagatorTest, TraceFlagsBufferGeneration)
{
  EXPECT_EQ(MapHttpTraceContext::TraceFlagsFromHex("00"), trace::TraceFlags());
}

TEST(TextMapPropagatorTest, NoSendEmptyTraceState)
{
  // If the trace state is empty, do not set the header.
  TextMapCarrierTest carrier;
  carrier.headers_ = {{"traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-0102030405060708-01"}};
  context::Context ctx1 = context::Context{
      trace::kSpanKey,
      nostd::shared_ptr<trace::Span>(new trace::DefaultSpan(trace::SpanContext::GetInvalid()))};
  context::Context ctx2 = format.Extract(carrier, ctx1);
  TextMapCarrierTest carrier2;
  format.Inject(carrier2, ctx2);
  EXPECT_TRUE(carrier2.headers_.count("traceparent") > 0);
  EXPECT_FALSE(carrier2.headers_.count("tracestate") > 0);
}

TEST(TextMapPropagatorTest, PropogateTraceState)
{
  TextMapCarrierTest carrier;
  carrier.headers_ = {{"traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-0102030405060708-01"},
                      {"tracestate", "congo=t61rcWkgMzE"}};
  context::Context ctx1 = context::Context{
      trace::kSpanKey,
      nostd::shared_ptr<trace::Span>(new trace::DefaultSpan(trace::SpanContext::GetInvalid()))};
  context::Context ctx2 = format.Extract(carrier, ctx1);

  TextMapCarrierTest carrier2;
  format.Inject(carrier2, ctx2);

  EXPECT_TRUE(carrier2.headers_.count("traceparent") > 0);
  EXPECT_TRUE(carrier2.headers_.count("tracestate") > 0);
  EXPECT_EQ(carrier2.headers_["tracestate"], "congo=t61rcWkgMzE");
}

TEST(TextMapPropagatorTest, PropagateInvalidContext)
{
  // Do not propagate invalid trace context.
  TextMapCarrierTest carrier;
  context::Context ctx{
      trace::kSpanKey,
      nostd::shared_ptr<trace::Span>(new trace::DefaultSpan(trace::SpanContext::GetInvalid()))};
  format.Inject(carrier, ctx);
  EXPECT_TRUE(carrier.headers_.count("traceparent") == 0);
  EXPECT_TRUE(carrier.headers_.count("tracestate") == 0);
}

TEST(TextMapPropagatorTest, SetRemoteSpan)
{
  TextMapCarrierTest carrier;
  carrier.headers_ = {{"traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-0102030405060708-01"}};
  context::Context ctx1 = context::Context{};
  context::Context ctx2 = format.Extract(carrier, ctx1);

  auto ctx2_span = ctx2.GetValue(trace::kSpanKey);
  EXPECT_TRUE(nostd::holds_alternative<nostd::shared_ptr<trace::Span>>(ctx2_span));

  auto span = nostd::get<nostd::shared_ptr<trace::Span>>(ctx2_span);

  EXPECT_EQ(Hex(span->GetContext().trace_id()), "4bf92f3577b34da6a3ce929d0e0e4736");
  EXPECT_EQ(Hex(span->GetContext().span_id()), "0102030405060708");
  EXPECT_EQ(span->GetContext().IsSampled(), true);
  EXPECT_EQ(span->GetContext().IsRemote(), true);
}

TEST(TextMapPropagatorTest, DoNotOverwriteContextWithInvalidSpan)
{
  TextMapCarrierTest carrier;
  constexpr uint8_t buf_span[]  = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr uint8_t buf_trace[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  trace::SpanContext span_context{trace::TraceId{buf_trace}, trace::SpanId{buf_span},
                                  trace::TraceFlags{true}, false};
  nostd::shared_ptr<trace::Span> sp{new trace::DefaultSpan{span_context}};

  // Make sure this invalid span does not overwrite the active span context
  carrier.headers_ = {{"traceparent", "00-FOO92f3577b34da6a3ce929d0e0e4736-010BAR0405060708-01"}};
  context::Context ctx1{trace::kSpanKey, sp};
  context::Context ctx2 = format.Extract(carrier, ctx1);
  auto ctx2_span        = ctx2.GetValue(trace::kSpanKey);
  auto span             = nostd::get<nostd::shared_ptr<trace::Span>>(ctx2_span);

  EXPECT_EQ(Hex(span->GetContext().trace_id()), "0102030405060708090a0b0c0d0e0f10");
}

TEST(TextMapPropagatorTest, GetCurrentSpan)
{
  TextMapCarrierTest carrier;
  constexpr uint8_t buf_span[]  = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr uint8_t buf_trace[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  auto trace_state = trace::TraceState::FromHeader("congo=t61rcWkgMzE");
  trace::SpanContext span_context{trace::TraceId{buf_trace}, trace::SpanId{buf_span},
                                  trace::TraceFlags{true}, false, trace_state};
  nostd::shared_ptr<trace::Span> sp{new trace::DefaultSpan{span_context}};

  // Set `sp` as the currently active span, which must be used by `Inject`.
  trace::Scope scoped_span{sp};

  format.Inject(carrier, context::RuntimeContext::GetCurrent());
  EXPECT_EQ(carrier.headers_["traceparent"],
            "00-0102030405060708090a0b0c0d0e0f10-0102030405060708-01");
  EXPECT_EQ(carrier.headers_["tracestate"], "congo=t61rcWkgMzE");
}

TEST(TextMapPropagatorTest, InvalidIdentitiesAreNotExtracted)
{
  TextMapCarrierTest carrier;
  std::vector<std::string> traces = {
      "ff-0af7651916cd43dd8448eb211c80319c-b9c7c989f97918e1-01",
      "00-0af7651916cd43dd8448eb211c80319c1-b9c7c989f97918e1-01",
      "00-0af7651916cd43dd8448eb211c80319c-b9c7c989f97918e11-01",
      "0-0af7651916cd43dd8448eb211c80319c-b9c7c989f97918e1-01",
      "00-0af7651916cd43dd8448eb211c80319c-b9c7c989f97918e1-0",
      "00-0af7651916cd43dd8448eb211c8031-b9c7c989f97918e1-01",
      "00-0af7651916cd43dd8448eb211c80319c-b9c7c989f97-01",
      "00-1-1-00",
      "00--b9c7c989f97918e1-01",
      "00-0af7651916cd43dd8448eb211c80319c1--01",
      "",
      "---",
  };

  for (auto &trace : traces)
  {
    carrier.headers_      = {{"traceparent", trace}};
    context::Context ctx1 = context::Context{};
    context::Context ctx2 = format.Extract(carrier, ctx1);

    auto span = trace::GetSpan(ctx2)->GetContext();
    EXPECT_FALSE(span.IsValid());
  }
}

TEST(GlobalTextMapPropagator, NoOpPropagator)
{

  auto propagator = context::propagation::GlobalTextMapPropagator::GetGlobalPropagator();
  TextMapCarrierTest carrier;

  carrier.headers_ = {{"traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-0102030405060708-01"},
                      {"tracestate", "congo=t61rcWkgMzE"}};
  context::Context ctx1 = context::Context{
      trace::kSpanKey,
      nostd::shared_ptr<trace::Span>(new trace::DefaultSpan(trace::SpanContext::GetInvalid()))};
  context::Context ctx2 = propagator->Extract(carrier, ctx1);

  TextMapCarrierTest carrier2;
  propagator->Inject(carrier2, ctx2);

  EXPECT_TRUE(carrier2.headers_.count("tracestate") == 0);
  EXPECT_TRUE(carrier2.headers_.count("traceparent") == 0);
}

TEST(GlobalPropagator, SetAndGet)
{

  auto trace_state_value = "congo=t61rcWkgMzE";
  context::propagation::GlobalTextMapPropagator::SetGlobalPropagator(
      nostd::shared_ptr<context::propagation::TextMapPropagator>(new MapHttpTraceContext()));

  auto propagator = context::propagation::GlobalTextMapPropagator::GetGlobalPropagator();

  TextMapCarrierTest carrier;
  carrier.headers_ = {{"traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-0102030405060708-01"},
                      {"tracestate", trace_state_value}};
  context::Context ctx1 = context::Context{
      trace::kSpanKey,
      nostd::shared_ptr<trace::Span>(new trace::DefaultSpan(trace::SpanContext::GetInvalid()))};
  context::Context ctx2 = propagator->Extract(carrier, ctx1);

  TextMapCarrierTest carrier2;
  propagator->Inject(carrier2, ctx2);

  EXPECT_TRUE(carrier.headers_.count("traceparent") > 0);
  EXPECT_TRUE(carrier.headers_.count("tracestate") > 0);
  EXPECT_EQ(carrier.headers_["tracestate"], trace_state_value);

  std::vector<std::string> fields;
  propagator->Fields([&fields](nostd::string_view field) {
    fields.push_back(field.data());
    return true;
  });
  EXPECT_EQ(fields.size(), 2);
  EXPECT_EQ(fields[0], trace::propagation::kTraceParent);
  EXPECT_EQ(fields[1], trace::propagation::kTraceState);
}
