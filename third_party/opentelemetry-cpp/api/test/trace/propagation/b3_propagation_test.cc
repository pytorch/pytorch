// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/trace/propagation/b3_propagator.h"
#include "opentelemetry/trace/scope.h"
#include "util.h"

#include <map>

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

using MapB3Context = trace::propagation::B3Propagator;

static MapB3Context format = MapB3Context();

using MapB3ContextMultiHeader = trace::propagation::B3PropagatorMultiHeader;

static MapB3ContextMultiHeader formatMultiHeader = MapB3ContextMultiHeader();

TEST(B3PropagationTest, TraceFlagsBufferGeneration)
{
  EXPECT_EQ(MapB3Context::TraceFlagsFromHex("0"), trace::TraceFlags());
  EXPECT_EQ(MapB3Context::TraceFlagsFromHex("1"), trace::TraceFlags(trace::TraceFlags::kIsSampled));
}

TEST(B3PropagationTest, PropagateInvalidContext)
{
  // Do not propagate invalid trace context.
  TextMapCarrierTest carrier;
  context::Context ctx{
      trace::kSpanKey,
      nostd::shared_ptr<trace::Span>(new trace::DefaultSpan(trace::SpanContext::GetInvalid()))};
  format.Inject(carrier, ctx);
  EXPECT_TRUE(carrier.headers_.count("b3") == 0);
}

TEST(B3PropagationTest, ExtractInvalidContext)
{
  TextMapCarrierTest carrier;
  carrier.headers_      = {{"b3", "00000000000000000000000000000000-0000000000000000-0"}};
  context::Context ctx1 = context::Context{};
  context::Context ctx2 = format.Extract(carrier, ctx1);
  auto ctx2_span        = ctx2.GetValue(trace::kSpanKey);
  EXPECT_FALSE(nostd::holds_alternative<nostd::shared_ptr<trace::Span>>(ctx2_span));
}

TEST(B3PropagationTest, DoNotOverwriteContextWithInvalidSpan)
{
  TextMapCarrierTest carrier;
  constexpr uint8_t buf_span[]  = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr uint8_t buf_trace[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  trace::SpanContext span_context{trace::TraceId{buf_trace}, trace::SpanId{buf_span},
                                  trace::TraceFlags{true}, false};
  nostd::shared_ptr<trace::Span> sp{new trace::DefaultSpan{span_context}};

  // Make sure this invalid span does not overwrite the active span context
  carrier.headers_ = {{"b3", "00000000000000000000000000000000-0000000000000000-0"}};
  context::Context ctx1{trace::kSpanKey, sp};
  context::Context ctx2 = format.Extract(carrier, ctx1);
  auto ctx2_span        = ctx2.GetValue(trace::kSpanKey);
  auto span             = nostd::get<nostd::shared_ptr<trace::Span>>(ctx2_span);

  EXPECT_EQ(Hex(span->GetContext().trace_id()), "0102030405060708090a0b0c0d0e0f10");
}

TEST(B3PropagationTest, DoNotExtractWithInvalidHex)
{
  TextMapCarrierTest carrier;
  carrier.headers_      = {{"b3", "0000000zzz0000000000000000000000-0000000zzz000000-1"}};
  context::Context ctx1 = context::Context{};
  context::Context ctx2 = format.Extract(carrier, ctx1);
  auto ctx2_span        = ctx2.GetValue(trace::kSpanKey);
  EXPECT_FALSE(nostd::holds_alternative<nostd::shared_ptr<trace::Span>>(ctx2_span));
}

TEST(B3PropagationTest, SetRemoteSpan)
{
  TextMapCarrierTest carrier;
  carrier.headers_ = {
      {"b3", "80f198ee56343ba864fe8b2a57d3eff7-e457b5a2e4d86bd1-1-05e3ac9a4f6e3b90"}};
  context::Context ctx1 = context::Context{};
  context::Context ctx2 = format.Extract(carrier, ctx1);

  auto ctx2_span = ctx2.GetValue(trace::kSpanKey);
  EXPECT_TRUE(nostd::holds_alternative<nostd::shared_ptr<trace::Span>>(ctx2_span));

  auto span = nostd::get<nostd::shared_ptr<trace::Span>>(ctx2_span);

  EXPECT_EQ(Hex(span->GetContext().trace_id()), "80f198ee56343ba864fe8b2a57d3eff7");
  EXPECT_EQ(Hex(span->GetContext().span_id()), "e457b5a2e4d86bd1");
  EXPECT_EQ(span->GetContext().IsSampled(), true);
  EXPECT_EQ(span->GetContext().IsRemote(), true);
}

TEST(B3PropagationTest, SetRemoteSpan_TraceIdShort)
{
  TextMapCarrierTest carrier;
  carrier.headers_      = {{"b3", "80f198ee56343ba8-e457b5a2e4d86bd1-1-05e3ac9a4f6e3b90"}};
  context::Context ctx1 = context::Context{};
  context::Context ctx2 = format.Extract(carrier, ctx1);

  auto ctx2_span = ctx2.GetValue(trace::kSpanKey);
  EXPECT_TRUE(nostd::holds_alternative<nostd::shared_ptr<trace::Span>>(ctx2_span));

  auto span = nostd::get<nostd::shared_ptr<trace::Span>>(ctx2_span);

  EXPECT_EQ(Hex(span->GetContext().trace_id()), "000000000000000080f198ee56343ba8");
  EXPECT_EQ(Hex(span->GetContext().span_id()), "e457b5a2e4d86bd1");
  EXPECT_EQ(span->GetContext().IsSampled(), true);
  EXPECT_EQ(span->GetContext().IsRemote(), true);
}

TEST(B3PropagationTest, SetRemoteSpan_SingleHeaderNoFlags)
{
  TextMapCarrierTest carrier;
  carrier.headers_      = {{"b3", "80f198ee56343ba864fe8b2a57d3eff7-e457b5a2e4d86bd1"}};
  context::Context ctx1 = context::Context{};
  context::Context ctx2 = format.Extract(carrier, ctx1);

  auto ctx2_span = ctx2.GetValue(trace::kSpanKey);
  EXPECT_TRUE(nostd::holds_alternative<nostd::shared_ptr<trace::Span>>(ctx2_span));

  auto span = nostd::get<nostd::shared_ptr<trace::Span>>(ctx2_span);

  EXPECT_EQ(Hex(span->GetContext().trace_id()), "80f198ee56343ba864fe8b2a57d3eff7");
  EXPECT_EQ(Hex(span->GetContext().span_id()), "e457b5a2e4d86bd1");
  EXPECT_EQ(span->GetContext().IsSampled(), false);
  EXPECT_EQ(span->GetContext().IsRemote(), true);
}

TEST(B3PropagationTest, SetRemoteSpanMultiHeader)
{
  TextMapCarrierTest carrier;
  carrier.headers_      = {{"X-B3-TraceId", "80f198ee56343ba864fe8b2a57d3eff7"},
                      {"X-B3-SpanId", "e457b5a2e4d86bd1"},
                      {"X-B3-Sampled", "1"}};
  context::Context ctx1 = context::Context{};
  context::Context ctx2 = format.Extract(carrier, ctx1);

  auto ctx2_span = ctx2.GetValue(trace::kSpanKey);
  EXPECT_TRUE(nostd::holds_alternative<nostd::shared_ptr<trace::Span>>(ctx2_span));

  auto span = nostd::get<nostd::shared_ptr<trace::Span>>(ctx2_span);

  EXPECT_EQ(Hex(span->GetContext().trace_id()), "80f198ee56343ba864fe8b2a57d3eff7");
  EXPECT_EQ(Hex(span->GetContext().span_id()), "e457b5a2e4d86bd1");
  EXPECT_EQ(span->GetContext().IsSampled(), true);
  EXPECT_EQ(span->GetContext().IsRemote(), true);
}

TEST(B3PropagationTest, GetCurrentSpan)
{
  TextMapCarrierTest carrier;
  constexpr uint8_t buf_span[]  = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr uint8_t buf_trace[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  trace::SpanContext span_context{trace::TraceId{buf_trace}, trace::SpanId{buf_span},
                                  trace::TraceFlags{true}, false};
  nostd::shared_ptr<trace::Span> sp{new trace::DefaultSpan{span_context}};

  // Set `sp` as the currently active span, which must be used by `Inject`.
  trace::Scope scoped_span{sp};

  format.Inject(carrier, context::RuntimeContext::GetCurrent());
  EXPECT_EQ(carrier.headers_["b3"], "0102030405060708090a0b0c0d0e0f10-0102030405060708-1");
}

TEST(B3PropagationTest, GetCurrentSpanMultiHeader)
{
  TextMapCarrierTest carrier;
  constexpr uint8_t buf_span[]  = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr uint8_t buf_trace[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  trace::SpanContext span_context{trace::TraceId{buf_trace}, trace::SpanId{buf_span},
                                  trace::TraceFlags{true}, false};
  nostd::shared_ptr<trace::Span> sp{new trace::DefaultSpan{span_context}};

  // Set `sp` as the currently active span, which must be used by `Inject`.
  trace::Scope scoped_span{sp};

  formatMultiHeader.Inject(carrier, context::RuntimeContext::GetCurrent());
  EXPECT_EQ(carrier.headers_["X-B3-TraceId"], "0102030405060708090a0b0c0d0e0f10");
  EXPECT_EQ(carrier.headers_["X-B3-SpanId"], "0102030405060708");
  EXPECT_EQ(carrier.headers_["X-B3-Sampled"], "1");
}
