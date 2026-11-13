/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "src/ApproximateClock.h"

#include <chrono>

#include <gtest/gtest.h>

using namespace libkineto;

TEST(ApproximateClockTest, ReturnsNonZero) {
  approx_time_t t = getApproximateTime();
  EXPECT_NE(t, 0);
}

TEST(ApproximateClockTest, IsMonotonic) {
  constexpr int kIterations = 1000;
  approx_time_t prev = getApproximateTime();
  for (int i = 0; i < kIterations; ++i) {
    approx_time_t curr = getApproximateTime();
    ASSERT_GE(curr, prev) << "Clock went backwards at iteration " << i;
    prev = curr;
  }
}

TEST(ApproximateClockTest, AdvancesOverTime) {
  approx_time_t t0 = getApproximateTime();

  // We use this pattern several times in these tests. We keep checking
  // approximate time until we find its value has increased, with a 10 ms
  // timeout. If we hit the timeout, the test fails. On platforms such as
  // x86, the first iteration should succeed as approximate time is tied
  // to the clock cycle. On some other platforms, such as aarch64, it may take
  // several clock cycles before approximate time actually increases.
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
  approx_time_t t1;
  do {
    t1 = getApproximateTime();
  } while (t1 == t0 && std::chrono::steady_clock::now() < deadline);
  EXPECT_GT(t1, t0);
}

TEST(ApproximateClockTest, MeasurePairCapturesBothClocks) {
  auto pair = ApproximateClockToUnixTimeConverter::measurePair();
  // Wall time should be a plausible epoch timestamp (after 2020, before 2100).
  constexpr libkineto::time_t kYear2020 = 1577836800LL * 1'000'000'000;
  constexpr libkineto::time_t kYear2100 = 4102444800LL * 1'000'000'000;
  EXPECT_GT(pair.t_, kYear2020);
  EXPECT_LT(pair.t_, kYear2100);
  EXPECT_NE(pair.approx_t_, 0) << "Approximate time should be non-zero";
}

TEST(ApproximateClockTest, ConverterProducesPlausibleEpochTime) {
  ApproximateClockToUnixTimeConverter converter;
  auto convert = converter.makeConverter();

  approx_time_t approx_now = getApproximateTime();
  auto converted_ns = convert(approx_now);

  constexpr libkineto::time_t kYear2020 = 1577836800LL * 1'000'000'000;
  constexpr libkineto::time_t kYear2100 = 4102444800LL * 1'000'000'000;
  EXPECT_GT(converted_ns, kYear2020);
  EXPECT_LT(converted_ns, kYear2100);
}

TEST(ApproximateClockTest, ConverterPreservesOrdering) {
  ApproximateClockToUnixTimeConverter converter;
  auto convert = converter.makeConverter();

  approx_time_t t0 = getApproximateTime();
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
  approx_time_t t1;
  do {
    t1 = getApproximateTime();
  } while (t1 == t0 && std::chrono::steady_clock::now() < deadline);

  auto wall0 = convert(t0);
  auto wall1 = convert(t1);
  EXPECT_GT(wall1, wall0) << "Converted times must preserve ordering";
}

TEST(ApproximateClockTest, GetTimeIsPositiveAndAdvances) {
  libkineto::time_t t0 = getTime();
  EXPECT_GT(t0, 0);
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
  libkineto::time_t t1;
  do {
    t1 = getTime();
  } while (t1 == t0 && std::chrono::steady_clock::now() < deadline);
  EXPECT_GT(t1, t0);
}
