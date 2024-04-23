// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/nostd/string_view.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "opentelemetry/baggage/baggage.h"

using namespace opentelemetry;
using namespace opentelemetry::baggage;

std::string header_with_custom_entries(size_t num_entries)
{
  std::string header;
  for (size_t i = 0; i < num_entries; i++)
  {
    std::string key   = "key" + std::to_string(i);
    std::string value = "value" + std::to_string(i);
    header += key + "=" + value;
    if (i != num_entries - 1)
    {
      header += ",";
    }
  }
  return header;
}

std::string header_with_custom_size(size_t key_value_size, size_t num_entries)
{
  std::string header = "";
  for (size_t i = 0; i < num_entries; i++)
  {
    std::string str = std::to_string(i + 1);
    str += "=";
    assert(key_value_size > str.size());
    for (size_t j = str.size(); j < key_value_size; j++)
    {
      str += "a";
    }

    header += str + ',';
  }

  header.pop_back();
  return header;
}

TEST(BaggageTest, ValidateExtractHeader)
{
  auto invalid_key_value_size_header = header_with_custom_size(Baggage::kMaxKeyValueSize + 5, 1);

  struct
  {
    const char *input;
    std::vector<const char *> keys;
    std::vector<const char *> values;
  } testcases[] = {
      {"k1=v1", {"k1"}, {"v1"}},
      {"k1=V1,K2=v2;metadata,k3=v3",
       {"k1", "K2", "k3"},
       {"V1", "v2;metadata", "v3"}},  // metadata is part of value
      {",k1 =v1,k2=v2 ; metadata,",
       {"k1", "k2"},
       {"v1", "v2; metadata"}},  // key and value are trimmed
      {"1a-2f%40foo=bar%251,a%2A%2Ffoo-_%2Fbar=bar+4",
       {"1a-2f@foo", "a*/foo-_/bar"},
       {"bar%1", "bar 4"}},                                       // decoding is done properly
      {"k1=v1,invalidmember,k2=v2", {"k1", "k2"}, {"v1", "v2"}},  // invalid member is skipped
      {",", {}, {}},
      {",=,", {}, {}},
      {"", {}, {}},
      {"k1=%5zv", {}, {}},  // invalid hex : invalid second digit
      {"k1=%5", {}, {}},    // invalid hex : missing two digits
      {"k%z2=v1", {}, {}},  // invalid hex : invalid first digit
      {"k%00=v1", {}, {}},  // key not valid
      {"k=v%7f", {}, {}},   // value not valid
      {invalid_key_value_size_header.data(), {}, {}}};
  for (auto &testcase : testcases)
  {
    auto baggage = Baggage::FromHeader(testcase.input);
    size_t index = 0;
    baggage->GetAllEntries([&testcase, &index](nostd::string_view key, nostd::string_view value) {
      EXPECT_EQ(key, testcase.keys[index]);
      EXPECT_EQ(value, testcase.values[index]);
      index++;
      return true;
    });
  }

  // For header with maximum threshold pairs, no pair is dropped
  auto max_pairs_header = header_with_custom_entries(Baggage::kMaxKeyValuePairs);
  EXPECT_EQ(Baggage::FromHeader(max_pairs_header.data())->ToHeader(), max_pairs_header.data());

  // Entries beyond threshold are dropped
  auto baggage = Baggage::FromHeader(header_with_custom_entries(Baggage::kMaxKeyValuePairs + 1));
  auto header  = baggage->ToHeader();
  common::KeyValueStringTokenizer kv_str_tokenizer(header);
  int expected_tokens = Baggage::kMaxKeyValuePairs;
  EXPECT_EQ(kv_str_tokenizer.NumTokens(), expected_tokens);

  // For header with total size more than threshold, baggage is empty
  int num_pairs_with_max_size = Baggage::kMaxSize / Baggage::kMaxKeyValueSize;
  auto invalid_total_size_header =
      header_with_custom_size(Baggage::kMaxKeyValueSize, num_pairs_with_max_size + 1);
  EXPECT_EQ(Baggage::FromHeader(invalid_total_size_header.data())->ToHeader(), "");
}

TEST(BaggageTest, ValidateInjectHeader)
{
  struct
  {
    std::vector<const char *> keys;
    std::vector<const char *> values;
    const char *header;
  } testcases[] = {{{"k1"}, {"v1"}, "k1=v1"},
                   {{"k3", "k2", "k1"}, {"", "v2", "v1"}, "k1=v1,k2=v2,k3="},  // empty value
                   {{"1a-2f@foo", "a*/foo-_/bar"},
                    {"bar%1", "bar 4"},
                    "a%2A%2Ffoo-_%2Fbar=bar+4,1a-2f%40foo=bar%251"},  // encoding is done properly
                   {{"foo 1"},
                    {"bar 1;  metadata ; ;;"},
                    "foo+1=bar+1;  metadata ; ;;"}};  // metadata is added without encoding

  for (auto &testcase : testcases)
  {
    nostd::shared_ptr<Baggage> baggage(new Baggage{});
    for (size_t i = 0; i < testcase.keys.size(); i++)
    {
      baggage = baggage->Set(testcase.keys[i], testcase.values[i]);
    }
    EXPECT_EQ(baggage->ToHeader(), testcase.header);
  }
}

TEST(BaggageTest, BaggageGet)
{
  auto header  = header_with_custom_entries(Baggage::kMaxKeyValuePairs);
  auto baggage = Baggage::FromHeader(header);

  std::string value;
  EXPECT_TRUE(baggage->GetValue("key0", value));
  EXPECT_EQ(value, "value0");
  EXPECT_TRUE(baggage->GetValue("key16", value));
  EXPECT_EQ(value, "value16");

  EXPECT_TRUE(baggage->GetValue("key31", value));
  EXPECT_EQ(value, "value31");

  EXPECT_FALSE(baggage->GetValue("key181", value));
}

TEST(BaggageTest, BaggageSet)
{
  std::string header = "k1=v1,k2=v2";
  auto baggage       = Baggage::FromHeader(header);

  std::string value;
  baggage = baggage->Set("k3", "v3");
  EXPECT_TRUE(baggage->GetValue("k3", value));
  EXPECT_EQ(value, "v3");

  baggage = baggage->Set("k3", "v3_1");  // key should be updated with the latest value
  EXPECT_TRUE(baggage->GetValue("k3", value));
  EXPECT_EQ(value, "v3_1");

  header  = header_with_custom_entries(Baggage::kMaxKeyValuePairs);
  baggage = Baggage::FromHeader(header);
  baggage = baggage->Set("key0", "0");  // updating on max list should work
  EXPECT_TRUE(baggage->GetValue("key0", value));
  EXPECT_EQ(value, "0");

  header  = "k1=v1,k2=v2";
  baggage = Baggage::FromHeader(header);
  baggage = baggage->Set("", "n_v1");  // adding invalid key, should return copy of same baggage
  EXPECT_EQ(baggage->ToHeader(), header);

  header  = "k1=v1,k2=v2";
  baggage = Baggage::FromHeader(header);
  baggage = baggage->Set("k1", "\x1A");  // adding invalid value, should return copy of same baggage
  EXPECT_EQ(baggage->ToHeader(), header);
}

TEST(BaggageTest, BaggageRemove)
{
  auto header  = header_with_custom_entries(Baggage::kMaxKeyValuePairs);
  auto baggage = Baggage::FromHeader(header);
  std::string value;

  // existing key is removed
  EXPECT_TRUE(baggage->GetValue("key0", value));
  auto new_baggage = baggage->Delete("key0");
  EXPECT_FALSE(new_baggage->GetValue("key0", value));

  // trying Delete on non existent key
  EXPECT_FALSE(baggage->GetValue("key181", value));
  auto new_baggage_2 = baggage->Delete("key181");
  EXPECT_FALSE(new_baggage_2->GetValue("key181", value));
}

TEST(BaggageTest, BaggageGetAll)
{
  std::string baggage_header           = "k1=v1,k2=v2,k3=v3";
  auto baggage                         = Baggage::FromHeader(baggage_header);
  const int kNumPairs                  = 3;
  nostd::string_view keys[kNumPairs]   = {"k1", "k2", "k3"};
  nostd::string_view values[kNumPairs] = {"v1", "v2", "v3"};
  size_t index                         = 0;
  baggage->GetAllEntries(
      [&keys, &values, &index](nostd::string_view key, nostd::string_view value) {
        EXPECT_EQ(key, keys[index]);
        EXPECT_EQ(value, values[index]);
        index++;
        return true;
      });
}
