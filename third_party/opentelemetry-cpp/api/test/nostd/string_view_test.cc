// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/nostd/string_view.h"

#include <gtest/gtest.h>
#include <cstring>
#include <map>

using opentelemetry::nostd::string_view;

TEST(StringViewTest, DefaultConstruction)
{
  string_view ref;
  EXPECT_EQ(ref.data(), nullptr);
  EXPECT_EQ(ref.length(), 0);
}

TEST(StringViewTest, CStringInitialization)
{
  const char *val = "hello world";

  string_view ref(val);

  EXPECT_EQ(ref.data(), val);
  EXPECT_EQ(ref.length(), std::strlen(val));
}

TEST(StringViewTest, StdStringInitialization)
{
  const std::string val = "hello world";

  string_view ref(val);

  EXPECT_EQ(ref.data(), val.data());
  EXPECT_EQ(ref.length(), val.size());
}

TEST(StringViewTest, Copy)
{
  const std::string val = "hello world";

  string_view ref(val);
  string_view cpy(ref);

  EXPECT_EQ(cpy.data(), val);
  EXPECT_EQ(cpy.length(), val.length());
  EXPECT_EQ(cpy, val);
}

TEST(StringViewTest, Accessor)
{
  string_view s = "abc123";
  EXPECT_EQ(s.data(), &s[0]);
  EXPECT_EQ(s.data() + 1, &s[1]);
}

TEST(StringViewTest, ExplicitStdStringConversion)
{
  std::string s = static_cast<std::string>(string_view{"abc"});
  EXPECT_EQ(s, "abc");
}

TEST(StringViewTest, SubstrPortion)
{
  string_view s = "abc123";
  EXPECT_EQ("123", s.substr(3));
  EXPECT_EQ("12", s.substr(3, 2));
}

TEST(StringViewTest, SubstrOutOfRange)
{
  string_view s = "abc123";
#if __EXCEPTIONS || (defined(OPENTELEMETRY_STL_VERSION) && (OPENTELEMETRY_STL_VERSION >= 2017))
  EXPECT_THROW((void)s.substr(10), std::out_of_range);
#else
  EXPECT_DEATH({ s.substr(10); }, "");
#endif
}

TEST(StringViewTest, FindSingleCharacter)
{
  string_view s = "abc";

  // starting from 0-th position (default)
  EXPECT_EQ(s.find('a'), 0);
  EXPECT_EQ(s.find('b'), 1);
  EXPECT_EQ(s.find('c'), 2);
  EXPECT_EQ(s.find('d'), -1);  // FIXME: string_view:npos - problem with linker

  // starting from given index
  EXPECT_EQ(s.find('a', 1), -1);
  EXPECT_EQ(s.find('b', 1), 1);

  // out of index
  EXPECT_EQ(s.find('a', 10), -1);
}

TEST(StringViewTest, Compare)
{
  string_view s1 = "aaa";
  string_view s2 = "bbb";
  string_view s3 = "aaa";

  // Equals
  EXPECT_EQ(s1, s3);
  EXPECT_EQ(s1, s1);

  // Less then
  EXPECT_LT(s1, s2);

  // Greater then
  EXPECT_GT(s2, s1);
}

TEST(StringViewTest, MapKeyOrdering)
{
  std::map<string_view, size_t> m = {{"bbb", 2}, {"aaa", 1}, {"ccc", 3}};
  size_t i                        = 1;
  for (const auto &kv : m)
  {
    EXPECT_EQ(kv.second, i);
    i++;
  }
}
