#include <c10/util/ApproximateClock.h>

#include <chrono>

#include <gtest/gtest.h>

using namespace c10;

TEST(ApproximateClockTest, ConverterHandlesTimestampBeforeCalibrationStart) {
  approx_time_t t_before = getApproximateTime();
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
  while (getApproximateTime() == t_before &&
         std::chrono::steady_clock::now() < deadline) {
  }

  ApproximateClockToUnixTimeConverter converter;
  auto convert = converter.makeConverter();
  approx_time_t t_after = getApproximateTime();

  auto wall_before = convert(t_before);
  auto wall_after = convert(t_after);
  constexpr c10::time_t kYear2020 = 1577836800LL * 1'000'000'000;
  EXPECT_GT(wall_before, kYear2020);
  EXPECT_LT(wall_before, wall_after);
  EXPECT_EQ(convert(0), 0);
}
