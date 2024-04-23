
// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/baggage/propagation/baggage_propagator.h"
#include <gtest/gtest.h>
#include <map>
#include <string>
#include "opentelemetry/baggage/baggage_context.h"

using namespace opentelemetry;
using namespace opentelemetry::baggage::propagation;

class BaggageCarrierTest : public context::propagation::TextMapCarrier
{
public:
  BaggageCarrierTest() = default;
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

static BaggagePropagator format;

TEST(BaggagePropagatorTest, ExtractNoBaggageHeader)
{
  BaggageCarrierTest carrier;
  carrier.headers_      = {};
  context::Context ctx1 = context::Context{};
  context::Context ctx2 = format.Extract(carrier, ctx1);
  auto ctx2_baggage     = baggage::GetBaggage(ctx2);
  EXPECT_EQ(ctx2_baggage->ToHeader(), "");
}

TEST(BaggagePropagatorTest, ExtractAndInjectBaggage)
{
  // create header string for baggage larger than allowed size (kMaxKeyValueSize)
  std::string very_large_baggage_header =
      std::string(baggage::Baggage::kMaxKeyValueSize / 2 + 1, 'k') + "=" +
      std::string(baggage::Baggage::kMaxKeyValueSize / 2 + 1, 'v');

  std::map<std::string, std::string> baggages = {
      {"key1=val1,key2=val2", "key1=val1,key2=val2"},                // valid header
      {"key1 =   val1,  key2 =val2   ", "key1=val1,key2=val2"},      // valid header with spaces
      {"key1=val1,key2=val2;prop=1", "key1=val1,key2=val2;prop=1"},  // valid header with properties
      {"key%2C1=val1,key2=val2%2Cval3",
       "key%2C1=val1,key2=val2%2Cval3"},                      // valid header with url escape
      {"key1=val1,key2=val2,a,val3", "key1=val1,key2=val2"},  // valid header with invalid value
      {"key1=,key2=val2", "key1=,key2=val2"},                 // valid header with empty value
      {"invalid_header", ""},                                 // invalid header
      {very_large_baggage_header, ""}};  // baggage header larger than allowed size.

  for (auto baggage : baggages)
  {
    BaggageCarrierTest carrier1;
    carrier1.headers_[baggage::kBaggageHeader.data()] = baggage.first;
    context::Context ctx1                             = context::Context{};
    context::Context ctx2                             = format.Extract(carrier1, ctx1);

    BaggageCarrierTest carrier2;
    format.Inject(carrier2, ctx2);
    EXPECT_EQ(carrier2.headers_[baggage::kBaggageHeader.data()], baggage.second);

    std::vector<std::string> fields;
    format.Fields([&fields](nostd::string_view field) {
      fields.push_back(field.data());
      return true;
    });
    EXPECT_EQ(fields.size(), 1);
    EXPECT_EQ(fields[0], baggage::kBaggageHeader.data());
  }
}

TEST(BaggagePropagatorTest, InjectEmptyHeader)
{
  // Test Missing baggage from context
  BaggageCarrierTest carrier;
  context::Context ctx = context::Context{};
  format.Inject(carrier, ctx);
  EXPECT_EQ(carrier.headers_.find(baggage::kBaggageHeader), carrier.headers_.end());

  {
    // Test empty baggage in context
    BaggageCarrierTest carrier1;
    carrier1.headers_[baggage::kBaggageHeader.data()] = "";
    context::Context ctx1                             = context::Context{};
    context::Context ctx2                             = format.Extract(carrier1, ctx1);
    format.Inject(carrier, ctx2);
    EXPECT_EQ(carrier.headers_.find(baggage::kBaggageHeader), carrier.headers_.end());
  }
  {
    // Invalid baggage in context
    BaggageCarrierTest carrier1;
    carrier1.headers_[baggage::kBaggageHeader.data()] = "InvalidBaggageData";
    context::Context ctx1                             = context::Context{};
    context::Context ctx2                             = format.Extract(carrier1, ctx1);

    format.Inject(carrier, ctx2);
    EXPECT_EQ(carrier.headers_.find(baggage::kBaggageHeader), carrier.headers_.end());
  }
}
