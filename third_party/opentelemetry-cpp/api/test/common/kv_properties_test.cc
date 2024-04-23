
// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <opentelemetry/common/kv_properties.h>

#include <string>
#include <utility>
#include <vector>

// ------------------------- Entry class tests ---------------------------------

using namespace opentelemetry;
using opentelemetry::common::KeyValueProperties;
// Test constructor that takes a key-value pair
TEST(EntryTest, KeyValueConstruction)
{
  opentelemetry::nostd::string_view key = "test_key";
  opentelemetry::nostd::string_view val = "test_value";
  KeyValueProperties::Entry e(key, val);

  EXPECT_EQ(key.size(), e.GetKey().size());
  EXPECT_EQ(key, e.GetKey());

  EXPECT_EQ(val.size(), e.GetValue().size());
  EXPECT_EQ(val, e.GetValue());
}

// Test copy constructor
TEST(EntryTest, Copy)
{
  KeyValueProperties::Entry e("test_key", "test_value");
  KeyValueProperties::Entry copy(e);
  EXPECT_EQ(copy.GetKey(), e.GetKey());
  EXPECT_EQ(copy.GetValue(), e.GetValue());
}

// Test assignment operator
TEST(EntryTest, Assignment)
{
  KeyValueProperties::Entry e("test_key", "test_value");
  KeyValueProperties::Entry empty;
  empty = e;
  EXPECT_EQ(empty.GetKey(), e.GetKey());
  EXPECT_EQ(empty.GetValue(), e.GetValue());
}

TEST(EntryTest, SetValue)
{
  KeyValueProperties::Entry e("test_key", "test_value");
  opentelemetry::nostd::string_view new_val = "new_value";
  e.SetValue(new_val);

  EXPECT_EQ(new_val.size(), e.GetValue().size());
  EXPECT_EQ(new_val, e.GetValue());
}

// ------------------------- KeyValueStringTokenizer tests ---------------------------------

using opentelemetry::common::KeyValueStringTokenizer;
using opentelemetry::common::KeyValueStringTokenizerOptions;

TEST(KVStringTokenizer, SinglePair)
{
  bool valid_kv;
  nostd::string_view key, value;
  opentelemetry::nostd::string_view str = "k1=v1";
  KeyValueStringTokenizerOptions opts;
  KeyValueStringTokenizer tk(str, opts);
  EXPECT_TRUE(tk.next(valid_kv, key, value));
  EXPECT_TRUE(valid_kv);
  EXPECT_EQ(key, "k1");
  EXPECT_EQ(value, "v1");
  EXPECT_FALSE(tk.next(valid_kv, key, value));
}

TEST(KVStringTokenizer, AcceptEmptyEntries)
{
  bool valid_kv;
  nostd::string_view key, value;
  opentelemetry::nostd::string_view str = ":k1=v1::k2=v2: ";
  KeyValueStringTokenizerOptions opts;
  opts.member_separator     = ':';
  opts.ignore_empty_members = false;

  KeyValueStringTokenizer tk(str, opts);
  EXPECT_TRUE(tk.next(valid_kv, key, value));  // empty pair
  EXPECT_TRUE(tk.next(valid_kv, key, value));
  EXPECT_TRUE(valid_kv);
  EXPECT_EQ(key, "k1");
  EXPECT_EQ(value, "v1");
  EXPECT_TRUE(tk.next(valid_kv, key, value));  // empty pair
  EXPECT_EQ(key, "");
  EXPECT_EQ(value, "");
  EXPECT_TRUE(tk.next(valid_kv, key, value));
  EXPECT_TRUE(tk.next(valid_kv, key, value));  // empty pair
  EXPECT_FALSE(tk.next(valid_kv, key, value));
}

TEST(KVStringTokenizer, ValidPairsWithEmptyEntries)
{
  opentelemetry::nostd::string_view str = "k1:v1===k2:v2==";
  bool valid_kv;
  nostd::string_view key, value;
  KeyValueStringTokenizerOptions opts;
  opts.member_separator    = '=';
  opts.key_value_separator = ':';

  KeyValueStringTokenizer tk(str, opts);
  EXPECT_TRUE(tk.next(valid_kv, key, value));
  EXPECT_TRUE(valid_kv);
  EXPECT_EQ(key, "k1");
  EXPECT_EQ(value, "v1");

  EXPECT_TRUE(tk.next(valid_kv, key, value));
  EXPECT_TRUE(valid_kv);
  EXPECT_EQ(key, "k2");
  EXPECT_EQ(value, "v2");

  EXPECT_FALSE(tk.next(valid_kv, key, value));
}

TEST(KVStringTokenizer, InvalidPairs)
{
  opentelemetry::nostd::string_view str = "k1=v1,invalid  ,,  k2=v2   ,invalid";
  KeyValueStringTokenizer tk(str);
  bool valid_kv;
  nostd::string_view key, value;
  EXPECT_TRUE(tk.next(valid_kv, key, value));

  EXPECT_TRUE(valid_kv);
  EXPECT_EQ(key, "k1");
  EXPECT_EQ(value, "v1");

  EXPECT_TRUE(tk.next(valid_kv, key, value));
  EXPECT_FALSE(valid_kv);

  EXPECT_TRUE(tk.next(valid_kv, key, value));
  EXPECT_TRUE(valid_kv);
  EXPECT_EQ(key, "k2");
  EXPECT_EQ(value, "v2");

  EXPECT_TRUE(tk.next(valid_kv, key, value));
  EXPECT_FALSE(valid_kv);

  EXPECT_FALSE(tk.next(valid_kv, key, value));
}

TEST(KVStringTokenizer, NumTokens)
{
  struct
  {
    const char *input;
    size_t expected;
  } testcases[] = {{"k1=v1", 1},
                   {" ", 1},
                   {"k1=v1,k2=v2,k3=v3", 3},
                   {"k1=v1,", 1},
                   {"k1=v1,k2=v2,invalidmember", 3},
                   {"", 0}};
  for (auto &testcase : testcases)
  {
    KeyValueStringTokenizer tk(testcase.input);
    EXPECT_EQ(tk.NumTokens(), testcase.expected);
  }
}

//------------------------- KeyValueProperties tests ---------------------------------

TEST(KeyValueProperties, PopulateKVIterableContainer)
{
  std::vector<std::pair<std::string, std::string>> kv_pairs = {{"k1", "v1"}, {"k2", "v2"}};

  auto kv_properties = KeyValueProperties(kv_pairs);
  EXPECT_EQ(kv_properties.Size(), 2);

  std::string value;
  bool present = kv_properties.GetValue("k1", value);
  EXPECT_TRUE(present);
  EXPECT_EQ(value, "v1");

  present = kv_properties.GetValue("k2", value);
  EXPECT_TRUE(present);
  EXPECT_EQ(value, "v2");
}

TEST(KeyValueProperties, AddEntry)
{
  auto kv_properties = KeyValueProperties(1);
  kv_properties.AddEntry("k1", "v1");
  std::string value;
  bool present = kv_properties.GetValue("k1", value);
  EXPECT_TRUE(present);
  EXPECT_EQ(value, "v1");

  kv_properties.AddEntry("k2", "v2");  // entry will not be added as max size reached.
  EXPECT_EQ(kv_properties.Size(), 1);
  present = kv_properties.GetValue("k2", value);
  EXPECT_FALSE(present);
}

TEST(KeyValueProperties, GetValue)
{
  auto kv_properties = KeyValueProperties(1);
  kv_properties.AddEntry("k1", "v1");
  std::string value;
  bool present = kv_properties.GetValue("k1", value);
  EXPECT_TRUE(present);
  EXPECT_EQ(value, "v1");

  present = kv_properties.GetValue("k3", value);
  EXPECT_FALSE(present);
}

TEST(KeyValueProperties, GetAllEntries)
{
  std::vector<std::pair<std::string, std::string>> kv_pairs = {
      {"k1", "v1"}, {"k2", "v2"}, {"k3", "v3"}};
  const size_t kNumPairs                              = 3;
  opentelemetry::nostd::string_view keys[kNumPairs]   = {"k1", "k2", "k3"};
  opentelemetry::nostd::string_view values[kNumPairs] = {"v1", "v2", "v3"};
  auto kv_properties                                  = KeyValueProperties(kv_pairs);

  size_t index = 0;
  kv_properties.GetAllEntries(
      [&keys, &values, &index](nostd::string_view key, nostd::string_view value) {
        EXPECT_EQ(key, keys[index]);
        EXPECT_EQ(value, values[index]);
        index++;
        return true;
      });

  EXPECT_EQ(index, kNumPairs);
}
