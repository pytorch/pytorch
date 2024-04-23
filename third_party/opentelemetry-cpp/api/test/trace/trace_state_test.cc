// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/trace/trace_state.h"

#include <gtest/gtest.h>
#include "opentelemetry/nostd/string_view.h"

namespace
{

using opentelemetry::trace::TraceState;
namespace nostd = opentelemetry::nostd;

// Random string of length 257. Used for testing strings with max length 256.
const char *kLongString =
    "4aekid3he76zgytjavudqqeltyvu5zqio2lx7d92dlxlf0z4883irvxuwelsq27sx1mlrjg3r7ad3jeq09rjppyd9veorg"
    "2nmihy4vilabfts8bsxruih0urusmjnglzl3iwpjinmo835dbojcrd73p56nw80v4xxrkye59ytmu5v84ysfa24d58ovv9"
    "w1n54n0mhhf4z0mpv6oudywrp9vfoks6lrvxv3uihvbi2ihazf237kvt1nbsjn3kdvfdb";

// -------------------------- TraceState class tests ---------------------------

std::string create_ts_return_header(std::string header)
{
  auto ts = TraceState::FromHeader(header);
  return ts->ToHeader();
}

std::string header_with_max_members()
{
  std::string header = "";
  auto max_members   = TraceState::kMaxKeyValuePairs;
  for (int i = 0; i < max_members; i++)
  {
    std::string key   = "key" + std::to_string(i);
    std::string value = "value" + std::to_string(i);
    header += key + "=" + value;
    if (i != max_members - 1)
    {
      header += ",";
    }
  }
  return header;
}

TEST(TraceStateTest, ValidateHeaderParsing)
{
  auto max_trace_state_header = header_with_max_members();

  struct
  {
    const char *input;
    const char *expected;
  } testcases[] = {{"k1=v1", "k1=v1"},
                   {"K1=V1", ""},
                   {"k1=v1,k2=v2,k3=v3", "k1=v1,k2=v2,k3=v3"},
                   {"k1=v1,k2=v2,,", "k1=v1,k2=v2"},
                   {"k1=v1,k2=v2,invalidmember", ""},
                   {"1a-2f@foo=bar1,a*/foo-_/bar=bar4", "1a-2f@foo=bar1,a*/foo-_/bar=bar4"},
                   {"1a-2f@foo=bar1,*/foo-_/bar=bar4", ""},
                   {",k1=v1", "k1=v1"},
                   {",", ""},
                   {",=,", ""},
                   {"", ""},
                   {max_trace_state_header.data(), max_trace_state_header.data()}};
  for (auto &testcase : testcases)
  {
    EXPECT_EQ(create_ts_return_header(testcase.input), testcase.expected);
  }
}

TEST(TraceStateTest, TraceStateGet)
{

  std::string trace_state_header = header_with_max_members();
  auto ts                        = TraceState::FromHeader(trace_state_header);

  std::string value;
  EXPECT_TRUE(ts->Get("key0", value));
  EXPECT_EQ(value, "value0");
  EXPECT_TRUE(ts->Get("key16", value));
  EXPECT_EQ(value, "value16");
  EXPECT_TRUE(ts->Get("key31", value));
  EXPECT_EQ(value, "value31");
  EXPECT_FALSE(ts->Get("key32", value));
}

TEST(TraceStateTest, TraceStateSet)
{
  std::string trace_state_header = "k1=v1,k2=v2";
  auto ts1                       = TraceState::FromHeader(trace_state_header);
  auto ts1_new                   = ts1->Set("k3", "v3");
  EXPECT_EQ(ts1_new->ToHeader(), "k3=v3,k1=v1,k2=v2");

  trace_state_header = header_with_max_members();
  auto ts2           = TraceState::FromHeader(trace_state_header);
  auto ts2_new =
      ts2->Set("n_k1", "n_v1");  // adding to max list, should return copy of existing list
  EXPECT_EQ(ts2_new->ToHeader(), trace_state_header);

  trace_state_header = "k1=v1,k2=v2";
  auto ts3           = TraceState::FromHeader(trace_state_header);
  auto ts3_new       = ts3->Set("*n_k1", "n_v1");  // adding invalid key, should return empty
  EXPECT_EQ(ts3_new->ToHeader(), "");
}

TEST(TraceStateTest, TraceStateDelete)
{
  std::string trace_state_header = "k1=v1,k2=v2,k3=v3";
  auto ts1                       = TraceState::FromHeader(trace_state_header);
  auto ts1_new                   = ts1->Delete(std::string("k1"));
  EXPECT_EQ(ts1_new->ToHeader(), "k2=v2,k3=v3");

  trace_state_header = "k1=v1";  // single list member
  auto ts2           = TraceState::FromHeader(trace_state_header);
  auto ts2_new       = ts2->Delete(std::string("k1"));
  EXPECT_EQ(ts2_new->ToHeader(), "");

  trace_state_header = "k1=v1";  // single list member, delete invalid entry
  auto ts3           = TraceState::FromHeader(trace_state_header);
  auto ts3_new       = ts3->Delete(std::string("InvalidKey"));
  EXPECT_EQ(ts3_new->ToHeader(), "");
}

TEST(TraceStateTest, Empty)
{
  std::string trace_state_header = "";
  auto ts                        = TraceState::FromHeader(trace_state_header);
  EXPECT_TRUE(ts->Empty());

  trace_state_header = "k1=v1,k2=v2";
  auto ts1           = TraceState::FromHeader(trace_state_header);
  EXPECT_FALSE(ts1->Empty());
}

TEST(TraceStateTest, GetAllEntries)
{
  std::string trace_state_header       = "k1=v1,k2=v2,k3=v3";
  auto ts1                             = TraceState::FromHeader(trace_state_header);
  const int kNumPairs                  = 3;
  nostd::string_view keys[kNumPairs]   = {"k1", "k2", "k3"};
  nostd::string_view values[kNumPairs] = {"v1", "v2", "v3"};
  size_t index                         = 0;
  ts1->GetAllEntries([&keys, &values, &index](nostd::string_view key, nostd::string_view value) {
    EXPECT_EQ(key, keys[index]);
    EXPECT_EQ(value, values[index]);
    index++;
    return true;
  });
}

TEST(TraceStateTest, IsValidKey)
{
  EXPECT_TRUE(TraceState::IsValidKey("valid-key23/*"));
  EXPECT_FALSE(TraceState::IsValidKey("Invalid_key"));
  EXPECT_FALSE(TraceState::IsValidKey("invalid$Key&"));
  EXPECT_FALSE(TraceState::IsValidKey(""));
  EXPECT_FALSE(TraceState::IsValidKey(kLongString));
}

TEST(TraceStateTest, IsValidValue)
{
  EXPECT_TRUE(TraceState::IsValidValue("valid-val$%&~"));
  EXPECT_FALSE(TraceState::IsValidValue("\tinvalid"));
  EXPECT_FALSE(TraceState::IsValidValue("invalid="));
  EXPECT_FALSE(TraceState::IsValidValue("invalid,val"));
  EXPECT_FALSE(TraceState::IsValidValue(""));
  EXPECT_FALSE(TraceState::IsValidValue(kLongString));
}

// Tests that keys and values don't depend on null terminators
TEST(TraceStateTest, MemorySafe)
{
  std::string trace_state_header       = "";
  auto ts                              = TraceState::FromHeader(trace_state_header);
  const int kNumPairs                  = 3;
  nostd::string_view key_string        = "test_key_1test_key_2test_key_3";
  nostd::string_view val_string        = "test_val_1test_val_2test_val_3";
  nostd::string_view keys[kNumPairs]   = {key_string.substr(0, 10), key_string.substr(10, 10),
                                        key_string.substr(20, 10)};
  nostd::string_view values[kNumPairs] = {val_string.substr(0, 10), val_string.substr(10, 10),
                                          val_string.substr(20, 10)};

  auto ts1     = ts->Set(keys[2], values[2]);
  auto ts2     = ts1->Set(keys[1], values[1]);
  auto ts3     = ts2->Set(keys[0], values[0]);
  size_t index = 0;

  ts3->GetAllEntries([&keys, &values, &index](nostd::string_view key, nostd::string_view value) {
    EXPECT_EQ(key, keys[index]);
    EXPECT_EQ(value, values[index]);
    index++;
    return true;
  });
}
}  // namespace
