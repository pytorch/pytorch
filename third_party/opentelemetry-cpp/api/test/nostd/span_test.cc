// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/nostd/span.h"

#include <cstdint>

#include <algorithm>
#include <array>
#include <iterator>
#include <list>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

using opentelemetry::nostd::span;

TEST(SpanTest, DefaultConstruction)
{
  span<int> s1;
  EXPECT_EQ(s1.data(), nullptr);
  EXPECT_EQ(s1.size(), 0);
  EXPECT_TRUE(s1.empty());

  span<int, 0> s2;
  EXPECT_EQ(s2.data(), nullptr);
  EXPECT_EQ(s2.size(), 0);
  EXPECT_TRUE(s2.empty());

  EXPECT_FALSE((std::is_default_constructible<span<int, 1>>::value));
}

TEST(SpanTest, Assignment)
{
  std::array<int, 3> array1 = {1, 2, 3};
  std::array<int, 3> array2 = {1, 2, 3};
  span<int> s1{array1.data(), array1.size()};
  span<int, 3> s2{array1.data(), array1.size()};

  span<int> s3;
  s3 = s1;
  EXPECT_EQ(s3.data(), array1.data());
  EXPECT_EQ(s3.size(), array1.size());

  span<int, 3> s4{array2};
  s4 = s2;
  EXPECT_EQ(s4.data(), array1.data());
  EXPECT_EQ(s4.size(), array1.size());
}

TEST(SpanTest, PointerCountConstruction)
{
  std::array<int, 3> array = {1, 2, 3};

  span<int> s1{array.data(), array.size()};
  EXPECT_EQ(s1.data(), array.data());
  EXPECT_EQ(s1.size(), array.size());

  span<int, 3> s2{array.data(), array.size()};
  EXPECT_EQ(s2.data(), array.data());
  EXPECT_EQ(s2.size(), array.size());
}

TEST(SpanTest, RangeConstruction)
{
  int array[] = {1, 2, 3};

  span<int> s1{std::begin(array), std::end(array)};
  EXPECT_EQ(s1.data(), array);
  EXPECT_EQ(s1.size(), 3);

  span<int, 3> s2{std::begin(array), std::end(array)};
  EXPECT_EQ(s2.data(), array);
  EXPECT_EQ(s2.size(), 3);
}

TEST(SpanTest, ArrayConstruction)
{
  int array1[]              = {1, 2, 3};
  std::array<int, 3> array2 = {1, 2, 3};

  span<int> s1{array1};
  EXPECT_EQ(s1.data(), array1);
  EXPECT_EQ(s1.size(), 3);

  span<int> s2{array2};
  EXPECT_EQ(s2.data(), array2.data());
  EXPECT_EQ(s2.size(), array2.size());

  span<int, 3> s3{array1};
  EXPECT_EQ(s3.data(), array1);
  EXPECT_EQ(s3.size(), 3);

  span<int, 3> s4{array2};
  EXPECT_EQ(s4.data(), array2.data());
  EXPECT_EQ(s4.size(), array2.size());

  EXPECT_FALSE((std::is_constructible<span<int, 2>, int(&)[3]>::value));
}

TEST(SpanTest, ContainerConstruction)
{
  std::vector<int> v = {1, 2, 3};

  span<int> s1{v};
  EXPECT_EQ(s1.data(), v.data());
  EXPECT_EQ(s1.size(), v.size());

  span<int, 3> s2{v.data(), 3};

  EXPECT_EQ(s2.data(), v.data());
  EXPECT_EQ(s2.size(), v.size());

  EXPECT_FALSE((std::is_constructible<span<int>, std::vector<double>>::value));
  EXPECT_FALSE((std::is_constructible<span<int>, std::list<int>>::value));
}

TEST(SpanTest, OtherSpanConstruction)
{
  std::array<int, 3> array = {1, 2, 3};
  span<int> s1{array.data(), array.size()};
  span<int, 3> s2{array.data(), array.size()};

  span<int> s3{s1};
  EXPECT_EQ(s3.data(), array.data());
  EXPECT_EQ(s3.size(), array.size());

  span<int> s4{s2};
  EXPECT_EQ(s4.data(), array.data());
  EXPECT_EQ(s4.size(), array.size());

  span<const int> s5{s1};
  EXPECT_EQ(s5.data(), array.data());
  EXPECT_EQ(s5.size(), array.size());

  EXPECT_FALSE((std::is_constructible<span<int>, span<const int>>::value));
  EXPECT_FALSE((std::is_constructible<span<int>, span<double>>::value));

  span<int, 3> s6{s2};
  EXPECT_EQ(s6.data(), array.data());
  EXPECT_EQ(s6.size(), array.size());

  span<const int, 3> s7{s2};
  EXPECT_EQ(s7.data(), array.data());
  EXPECT_EQ(s7.size(), array.size());

  EXPECT_FALSE((std::is_constructible<span<int, 3>, span<int, 4>>::value));
  EXPECT_FALSE((std::is_constructible<span<int, 3>, span<double, 3>>::value));
}

TEST(SpanTest, BracketOperator)
{
  std::array<int, 2> array = {1, 2};

  span<int> s1{array.data(), array.size()};
  EXPECT_EQ(s1[0], 1);
  EXPECT_EQ(s1[1], 2);

  span<int, 2> s2{array.data(), array.size()};
  EXPECT_EQ(s2[0], 1);
  EXPECT_EQ(s2[1], 2);
}

TEST(SpanTest, Iteration)
{
  std::array<int, 3> array = {1, 2, 3};

  span<int> s1{array.data(), array.size()};
  EXPECT_EQ(std::distance(s1.begin(), s1.end()), (ptrdiff_t)array.size());
  EXPECT_TRUE(std::equal(s1.begin(), s1.end(), array.begin()));

  span<int, 3> s2{array.data(), array.size()};
  EXPECT_EQ(std::distance(s2.begin(), s2.end()), (ptrdiff_t)array.size());
  EXPECT_TRUE(std::equal(s2.begin(), s2.end(), array.begin()));
}
