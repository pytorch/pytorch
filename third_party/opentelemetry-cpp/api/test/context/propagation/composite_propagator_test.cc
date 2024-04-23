// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"
#include "opentelemetry/trace/span_context.h"

#include "opentelemetry/context/propagation/composite_propagator.h"
#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/trace/default_span.h"
#include "opentelemetry/trace/propagation/b3_propagator.h"
#include "opentelemetry/trace/propagation/http_trace_context.h"

#include <map>
#include <memory>
#include <string>

#include <gtest/gtest.h>

using namespace opentelemetry;

template <typename T>
static std::string Hex(const T &id_item)
{
  char buf[T::kSize * 2];
  id_item.ToLowerBase16(buf);
  return std::string(buf, sizeof(buf));
}

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

class CompositePropagatorTest : public ::testing::Test
{

public:
  CompositePropagatorTest()
  {
    std::vector<std::unique_ptr<context::propagation::TextMapPropagator>> propogator_list = {};
    std::unique_ptr<context::propagation::TextMapPropagator> w3c_propogator(
        new trace::propagation::HttpTraceContext());
    std::unique_ptr<context::propagation::TextMapPropagator> b3_propogator(
        new trace::propagation::B3Propagator());
    propogator_list.push_back(std::move(w3c_propogator));
    propogator_list.push_back(std::move(b3_propogator));

    composite_propagator_ =
        new context::propagation::CompositePropagator(std::move(propogator_list));
  }

  ~CompositePropagatorTest() { delete composite_propagator_; }

protected:
  context::propagation::CompositePropagator *composite_propagator_;
};

TEST_F(CompositePropagatorTest, Extract)
{
  TextMapCarrierTest carrier;
  carrier.headers_ = {
      {"traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-0102030405060708-01"},
      {"b3", "80f198ee56343ba864fe8b2a57d3eff7-e457b5a2e4d86bd1-1-05e3ac9a4f6e3b90"}};
  context::Context ctx1 = context::Context{};

  context::Context ctx2 = composite_propagator_->Extract(carrier, ctx1);

  auto ctx2_span = ctx2.GetValue(trace::kSpanKey);
  EXPECT_TRUE(nostd::holds_alternative<nostd::shared_ptr<trace::Span>>(ctx2_span));

  auto span = nostd::get<nostd::shared_ptr<trace::Span>>(ctx2_span);

  // confirm last propagator in composite propagator list (B3 here) wins for same key
  // ("active_span" here).
  EXPECT_EQ(Hex(span->GetContext().trace_id()), "80f198ee56343ba864fe8b2a57d3eff7");
  EXPECT_EQ(Hex(span->GetContext().span_id()), "e457b5a2e4d86bd1");
  EXPECT_EQ(span->GetContext().IsSampled(), true);
  EXPECT_EQ(span->GetContext().IsRemote(), true);

  // Now check that last propagator does not win if there is no header for it
  carrier.headers_ = {{"traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-0102030405060708-00"}};
  ctx1             = context::Context{};

  ctx2 = composite_propagator_->Extract(carrier, ctx1);

  ctx2_span = ctx2.GetValue(trace::kSpanKey);
  EXPECT_TRUE(nostd::holds_alternative<nostd::shared_ptr<trace::Span>>(ctx2_span));

  span = nostd::get<nostd::shared_ptr<trace::Span>>(ctx2_span);

  // Here the first propagator (W3C) wins
  EXPECT_EQ(Hex(span->GetContext().trace_id()), "4bf92f3577b34da6a3ce929d0e0e4736");
  EXPECT_EQ(Hex(span->GetContext().span_id()), "0102030405060708");
  EXPECT_EQ(span->GetContext().IsSampled(), false);
  EXPECT_EQ(span->GetContext().IsRemote(), true);
}

TEST_F(CompositePropagatorTest, Inject)
{
  TextMapCarrierTest carrier;
  constexpr uint8_t buf_span[]  = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr uint8_t buf_trace[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  trace::SpanContext span_context{trace::TraceId{buf_trace}, trace::SpanId{buf_span},
                                  trace::TraceFlags{true}, false};
  nostd::shared_ptr<trace::Span> sp{new trace::DefaultSpan{span_context}};

  // Set `sp` as the currently active span, which must be used by `Inject`.
  trace::Scope scoped_span{sp};

  composite_propagator_->Inject(carrier, context::RuntimeContext::GetCurrent());
  EXPECT_EQ(carrier.headers_["traceparent"],
            "00-0102030405060708090a0b0c0d0e0f10-0102030405060708-01");
  EXPECT_EQ(carrier.headers_["b3"], "0102030405060708090a0b0c0d0e0f10-0102030405060708-1");

  std::vector<std::string> fields;
  composite_propagator_->Fields([&fields](nostd::string_view field) {
    fields.push_back(field.data());
    return true;
  });
  EXPECT_EQ(fields.size(), 3);
  EXPECT_EQ(fields[0], trace::propagation::kTraceParent);
  EXPECT_EQ(fields[1], trace::propagation::kTraceState);
  EXPECT_EQ(fields[2], trace::propagation::kB3CombinedHeader);
}
