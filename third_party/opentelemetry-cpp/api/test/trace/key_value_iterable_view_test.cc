// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/common/key_value_iterable_view.h"

#include <gtest/gtest.h>
#include <map>
#include "opentelemetry/nostd/type_traits.h"

using namespace opentelemetry;

static int TakeKeyValues(const common::KeyValueIterable &iterable)
{
  std::map<std::string, common::AttributeValue> result;
  int count = 0;
  iterable.ForEachKeyValue(
      [&](nostd::string_view /* key */, common::AttributeValue /* value */) noexcept {
        ++count;
        return true;
      });
  return count;
}

template <class T, nostd::enable_if_t<common::detail::is_key_value_iterable<T>::value> * = nullptr>
static int TakeKeyValues(const T &iterable)
{
  return TakeKeyValues(common::KeyValueIterableView<T>{iterable});
}

TEST(KeyValueIterableViewTest, is_key_value_iterable)
{
  using M1 = std::map<std::string, std::string>;
  EXPECT_TRUE(bool{common::detail::is_key_value_iterable<M1>::value});

  using M2 = std::map<std::string, int>;
  EXPECT_TRUE(bool{common::detail::is_key_value_iterable<M2>::value});

  using M3 = std::map<std::string, common::AttributeValue>;
  EXPECT_TRUE(bool{common::detail::is_key_value_iterable<M3>::value});

  struct A
  {};
  using M4 = std::map<std::string, A>;
  EXPECT_FALSE(bool{common::detail::is_key_value_iterable<M4>::value});
}

TEST(KeyValueIterableViewTest, ForEachKeyValue)
{
  std::map<std::string, std::string> m1 = {{"abc", "123"}, {"xyz", "456"}};
  EXPECT_EQ(TakeKeyValues(m1), 2);

  std::vector<std::pair<std::string, int>> v1 = {{"abc", 123}, {"xyz", 456}};
  EXPECT_EQ(TakeKeyValues(v1), 2);
}

TEST(KeyValueIterableViewTest, ForEachKeyValueWithExit)
{
  using M = std::map<std::string, std::string>;
  M m1    = {{"abc", "123"}, {"xyz", "456"}};
  common::KeyValueIterableView<M> iterable{m1};
  int count = 0;
  auto exit = iterable.ForEachKeyValue(
      [&count](nostd::string_view /*key*/, common::AttributeValue /*value*/) noexcept {
        ++count;
        return false;
      });
  EXPECT_EQ(count, 1);
  EXPECT_FALSE(exit);
}
