// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <c10/util/generic_math.h>

#include <gtest/gtest.h>

#include <cmath>

using namespace ::testing;

TEST(GenericMathTest, div_floor_test) {
  EXPECT_EQ(c10::div_floor_floating(5., 0.), INFINITY);
  EXPECT_DOUBLE_EQ(c10::div_floor_floating(5., 2.), 2.);
  EXPECT_DOUBLE_EQ(c10::div_floor_floating(5., -2.), -3.);
  EXPECT_EQ(c10::div_floor_integer(5, 2), 2);
  EXPECT_EQ(c10::div_floor_integer(5, -2), -3);
}
