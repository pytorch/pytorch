// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/common/timestamp.h"

#include <gtest/gtest.h>

using opentelemetry::common::SteadyTimestamp;
using opentelemetry::common::SystemTimestamp;

template <class Timestamp>
static bool AreNearlyEqual(const Timestamp &t1, const Timestamp &t2) noexcept
{
  return std::abs(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) < 2;
}

TEST(SystemTimestampTest, Construction)
{
  auto now_system = std::chrono::system_clock::now();

  SystemTimestamp t1;
  EXPECT_EQ(t1.time_since_epoch(), std::chrono::nanoseconds{0});

  SystemTimestamp t2{now_system};
  EXPECT_TRUE(AreNearlyEqual(now_system, static_cast<std::chrono::system_clock::time_point>(t2)));
  EXPECT_EQ(std::chrono::duration_cast<std::chrono::nanoseconds>(now_system.time_since_epoch()),
            t2.time_since_epoch());
}

TEST(SystemTimestampTest, Comparison)
{
  SystemTimestamp t1;
  SystemTimestamp t2;
  SystemTimestamp t3{std::chrono::nanoseconds{2}};

  EXPECT_EQ(t1, t1);
  EXPECT_EQ(t1, t2);
  EXPECT_EQ(t2, t1);
  EXPECT_NE(t1, t3);
  EXPECT_NE(t3, t1);
}

TEST(SteadyTimestampTest, Construction)
{
  auto now_steady = std::chrono::steady_clock::now();

  SteadyTimestamp t1;
  EXPECT_EQ(t1.time_since_epoch(), std::chrono::nanoseconds{0});

  SteadyTimestamp t2{now_steady};
  EXPECT_TRUE(AreNearlyEqual(now_steady, static_cast<std::chrono::steady_clock::time_point>(t2)));
  EXPECT_EQ(std::chrono::duration_cast<std::chrono::nanoseconds>(now_steady.time_since_epoch()),
            t2.time_since_epoch());
}

TEST(SteadyTimestampTest, Comparison)
{
  SteadyTimestamp t1;
  SteadyTimestamp t2;
  SteadyTimestamp t3{std::chrono::nanoseconds{2}};

  EXPECT_EQ(t1, t1);
  EXPECT_EQ(t1, t2);
  EXPECT_EQ(t2, t1);
  EXPECT_NE(t1, t3);
  EXPECT_NE(t3, t1);
}
