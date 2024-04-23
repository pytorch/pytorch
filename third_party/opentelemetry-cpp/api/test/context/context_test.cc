// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/context/context.h"

#include <map>

#include <gtest/gtest.h>

using namespace opentelemetry;

// Tests that the context constructor accepts an std::map.
TEST(ContextTest, ContextIterableAcceptsMap)
{
  std::map<std::string, context::ContextValue> map_test = {{"test_key", (int64_t)123}};
  context::Context test_context                         = context::Context(map_test);
}

// Tests that the GetValue method returns the expected value.
TEST(ContextTest, ContextGetValueReturnsExpectedValue)
{
  std::map<std::string, context::ContextValue> map_test = {{"test_key", (int64_t)123},
                                                           {"foo_key", (int64_t)456}};
  const context::Context test_context                   = context::Context(map_test);
  EXPECT_EQ(nostd::get<int64_t>(test_context.GetValue("test_key")), 123);
  EXPECT_EQ(nostd::get<int64_t>(test_context.GetValue("foo_key")), 456);
}

// Tests that the SetValues method accepts an std::map.
TEST(ContextTest, ContextSetValuesAcceptsMap)
{
  std::map<std::string, context::ContextValue> map_test       = {{"test_key", (int64_t)123}};
  std::map<std::string, context::ContextValue> map_test_write = {{"foo_key", (int64_t)456}};

  context::Context test_context = context::Context(map_test);
  context::Context foo_context  = test_context.SetValues(map_test_write);

  EXPECT_EQ(nostd::get<int64_t>(foo_context.GetValue("test_key")), 123);
  EXPECT_EQ(nostd::get<int64_t>(foo_context.GetValue("foo_key")), 456);
}

// Tests that the SetValues method accepts a nostd::string_view and
// context::ContextValue.
TEST(ContextTest, ContextSetValuesAcceptsStringViewContextValue)
{
  nostd::string_view string_view_test      = "string_view";
  context::ContextValue context_value_test = (int64_t)123;

  context::Context test_context = context::Context(string_view_test, context_value_test);
  context::Context foo_context  = test_context.SetValue(string_view_test, context_value_test);

  EXPECT_EQ(nostd::get<int64_t>(foo_context.GetValue(string_view_test)), 123);
}

// Tests that the original context does not change when a value is
// written to it.
TEST(ContextTest, ContextImmutability)
{
  std::map<std::string, context::ContextValue> map_test = {{"test_key", (int64_t)123}};

  context::Context context_test = context::Context(map_test);
  context::Context context_foo  = context_test.SetValue("foo_key", (int64_t)456);

  EXPECT_FALSE(nostd::holds_alternative<int64_t>(context_test.GetValue("foo_key")));
}

// Tests that writing the same to a context overwrites the original value.
TEST(ContextTest, ContextKeyOverwrite)
{
  std::map<std::string, context::ContextValue> map_test = {{"test_key", (int64_t)123}};

  context::Context context_test = context::Context(map_test);
  context::Context context_foo  = context_test.SetValue("test_key", (int64_t)456);

  EXPECT_EQ(nostd::get<int64_t>(context_foo.GetValue("test_key")), 456);
}

// Tests that the new Context Objects inherits the keys and values
// of the original context object.
TEST(ContextTest, ContextInheritance)
{
  using M = std::map<std::string, context::ContextValue>;

  M m1 = {{"test_key", (int64_t)123}, {"foo_key", (int64_t)321}};
  M m2 = {{"other_key", (int64_t)789}, {"another_key", (int64_t)987}};

  context::Context test_context = context::Context(m1);
  context::Context foo_context  = test_context.SetValues(m2);

  EXPECT_EQ(nostd::get<int64_t>(foo_context.GetValue("test_key")), 123);
  EXPECT_EQ(nostd::get<int64_t>(foo_context.GetValue("foo_key")), 321);
  EXPECT_EQ(nostd::get<int64_t>(foo_context.GetValue("other_key")), 789);
  EXPECT_EQ(nostd::get<int64_t>(foo_context.GetValue("another_key")), 987);

  EXPECT_TRUE(nostd::holds_alternative<nostd::monostate>(test_context.GetValue("other_key")));
  EXPECT_TRUE(nostd::holds_alternative<nostd::monostate>(test_context.GetValue("another_key")));
}

// Tests that copying a context copies the key value pairs as expected.
TEST(ContextTest, ContextCopyOperator)
{
  std::map<std::string, context::ContextValue> test_map = {
      {"test_key", (int64_t)123}, {"foo_key", (int64_t)456}, {"other_key", (int64_t)789}};

  context::Context test_context   = context::Context(test_map);
  context::Context copied_context = test_context;

  EXPECT_EQ(nostd::get<int64_t>(copied_context.GetValue("test_key")), 123);
  EXPECT_EQ(nostd::get<int64_t>(copied_context.GetValue("foo_key")), 456);
  EXPECT_EQ(nostd::get<int64_t>(copied_context.GetValue("other_key")), 789);
}

// Tests that the Context accepts an empty map.
TEST(ContextTest, ContextEmptyMap)
{
  std::map<std::string, context::ContextValue> map_test = {};
  context::Context test_context                         = context::Context(map_test);
}

// Tests that if a key exists within a context has key will return true
// false if not.
TEST(ContextTest, ContextHasKey)
{
  std::map<std::string, context::ContextValue> map_test = {{"test_key", (int64_t)123}};
  const context::Context context_test                   = context::Context(map_test);
  EXPECT_TRUE(context_test.HasKey("test_key"));
  EXPECT_FALSE(context_test.HasKey("foo_key"));
}

// Tests that a copied context returns true when compared
TEST(ContextTest, ContextCopyCompare)
{
  std::map<std::string, context::ContextValue> map_test = {{"test_key", (int64_t)123}};
  context::Context context_test                         = context::Context(map_test);
  context::Context copied_test                          = context_test;
  EXPECT_TRUE(context_test == copied_test);
}

// Tests that two differently constructed contexts return false when compared
TEST(ContextTest, ContextDiffCompare)
{
  std::map<std::string, context::ContextValue> map_test = {{"test_key", (int64_t)123}};
  std::map<std::string, context::ContextValue> map_foo  = {{"foo_key", (int64_t)123}};
  context::Context context_test                         = context::Context(map_test);
  context::Context foo_test                             = context::Context(map_foo);
  EXPECT_FALSE(context_test == foo_test);
}
