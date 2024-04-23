// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/trace/propagation/jaeger.h"
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

using Propagator = trace::propagation::JaegerPropagator;

static Propagator format = Propagator();

TEST(JaegerPropagatorTest, ExtractValidSpans)
{
  struct TestTrace
  {
    std::string trace_state;
    std::string expected_trace_id;
    std::string expected_span_id;
    bool sampled;
  };

  std::vector<TestTrace> traces = {
      {
          "4bf92f3577b34da6a3ce929d0e0e4736:0102030405060708:0:00",
          "4bf92f3577b34da6a3ce929d0e0e4736",
          "0102030405060708",
          false,
      },
      {
          "4bf92f3577b34da6a3ce929d0e0e4736:0102030405060708:0:ff",
          "4bf92f3577b34da6a3ce929d0e0e4736",
          "0102030405060708",
          true,
      },
      {
          "4bf92f3577b34da6a3ce929d0e0e4736:0102030405060708:0:f",
          "4bf92f3577b34da6a3ce929d0e0e4736",
          "0102030405060708",
          true,
      },
      {
          "a3ce929d0e0e4736:0102030405060708:0:00",
          "0000000000000000a3ce929d0e0e4736",
          "0102030405060708",
          false,
      },
      {
          "A3CE929D0E0E4736:ABCDEFABCDEF1234:0:01",
          "0000000000000000a3ce929d0e0e4736",
          "abcdefabcdef1234",
          true,
      },
      {
          "ff:ABCDEFABCDEF1234:0:0",
          "000000000000000000000000000000ff",
          "abcdefabcdef1234",
          false,
      },
      {
          "4bf92f3577b34da6a3ce929d0e0e4736:0102030405060708:0102030405060708:00",
          "4bf92f3577b34da6a3ce929d0e0e4736",
          "0102030405060708",
          false,
      },

  };

  for (TestTrace &test_trace : traces)
  {
    TextMapCarrierTest carrier;
    carrier.headers_      = {{"uber-trace-id", test_trace.trace_state}};
    context::Context ctx1 = context::Context{};
    context::Context ctx2 = format.Extract(carrier, ctx1);

    auto span = trace::GetSpan(ctx2)->GetContext();
    EXPECT_TRUE(span.IsValid());

    EXPECT_EQ(Hex(span.trace_id()), test_trace.expected_trace_id);
    EXPECT_EQ(Hex(span.span_id()), test_trace.expected_span_id);
    EXPECT_EQ(span.IsSampled(), test_trace.sampled);
    EXPECT_EQ(span.IsRemote(), true);
  }
}

TEST(JaegerPropagatorTest, ExctractInvalidSpans)
{
  TextMapCarrierTest carrier;
  std::vector<std::string> traces = {
      "4bf92f3577b34da6a3ce929d0e0e47344:0102030405060708:0:00",  // too long trace id
      "4bf92f3577b34da6a3ce929d0e0e4734:01020304050607089:0:00",  // too long span id
      "4bf92f3577b34da6x3ce929d0y0e4734:01020304050607089:0:00",  // invalid trace id character
      "4bf92f3577b34da6a3ce929d0e0e4734:01020304g50607089:0:00",  // invalid span id character
      "4bf92f3577b34da6a3ce929d0e0e4734::0:00",
      "",
      "::::",
      "0:0:0:0",
      ":abcdef12:0:0",
  };

  for (auto &trace : traces)
  {
    carrier.headers_      = {{"uber-trace-id", trace}};
    context::Context ctx1 = context::Context{};
    context::Context ctx2 = format.Extract(carrier, ctx1);

    auto span = trace::GetSpan(ctx2)->GetContext();
    EXPECT_FALSE(span.IsValid());
  }
}

TEST(JaegerPropagatorTest, DoNotOverwriteContextWithInvalidSpan)
{
  TextMapCarrierTest carrier;
  constexpr uint8_t buf_span[]  = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr uint8_t buf_trace[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  trace::SpanContext span_context{trace::TraceId{buf_trace}, trace::SpanId{buf_span},
                                  trace::TraceFlags{true}, false};
  nostd::shared_ptr<trace::Span> sp{new trace::DefaultSpan{span_context}};

  // Make sure this invalid span does not overwrite the active span context
  carrier.headers_ = {{"uber-trace-id", "foo:bar:0:00"}};
  context::Context ctx1{trace::kSpanKey, sp};
  context::Context ctx2 = format.Extract(carrier, ctx1);
  auto ctx2_span        = ctx2.GetValue(trace::kSpanKey);
  auto span             = nostd::get<nostd::shared_ptr<trace::Span>>(ctx2_span);

  EXPECT_EQ(Hex(span->GetContext().trace_id()), "0102030405060708090a0b0c0d0e0f10");
}

TEST(JaegerPropagatorTest, InjectsContext)
{
  TextMapCarrierTest carrier;
  constexpr uint8_t buf_span[]  = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr uint8_t buf_trace[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  trace::SpanContext span_context{trace::TraceId{buf_trace}, trace::SpanId{buf_span},
                                  trace::TraceFlags{true}, false};
  nostd::shared_ptr<trace::Span> sp{new trace::DefaultSpan{span_context}};
  trace::Scope scoped_span{sp};

  format.Inject(carrier, context::RuntimeContext::GetCurrent());
  EXPECT_EQ(carrier.headers_["uber-trace-id"],
            "0102030405060708090a0b0c0d0e0f10:0102030405060708:0:01");

  std::vector<std::string> fields;
  format.Fields([&fields](nostd::string_view field) {
    fields.push_back(field.data());
    return true;
  });
  EXPECT_EQ(fields.size(), 1);
  EXPECT_EQ(fields[0], opentelemetry::trace::propagation::kJaegerTraceHeader);
}

TEST(JaegerPropagatorTest, DoNotInjectInvalidContext)
{
  TextMapCarrierTest carrier;
  context::Context ctx{
      trace::kSpanKey,
      nostd::shared_ptr<trace::Span>(new trace::DefaultSpan(trace::SpanContext::GetInvalid()))};
  format.Inject(carrier, ctx);
  EXPECT_TRUE(carrier.headers_.count("uber-trace-id") == 0);
}
