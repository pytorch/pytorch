#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <c10/util/ApproximateClock.h>
#include <c10/util/irange.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/util.h>

TEST(ProfilerTest, AppendOnlyList) {
  const int n = 4096;
  torch::profiler::impl::AppendOnlyList<int, 1024> list;
  for (const auto i : c10::irange(n)) {
    list.emplace_back(i);
    ASSERT_EQ(list.size(), i + 1);
  }

  int expected = 0;
  for (const auto i : list) {
    ASSERT_EQ(i, expected++);
  }
  ASSERT_EQ(expected, n);

  list.clear();
  ASSERT_EQ(list.size(), 0);
}

TEST(ProfilerTest, AppendOnlyList_ref) {
  const int n = 512;
  torch::profiler::impl::AppendOnlyList<std::pair<int, int>, 64> list;
  std::vector<std::pair<int, int>*> refs;
  for (const auto _ : c10::irange(n)) {
    refs.push_back(list.emplace_back());
  }

  for (const auto i : c10::irange(n)) {
    *refs.at(i) = {i, 0};
  }

  int expected = 0;
  for (const auto& i : list) {
    ASSERT_EQ(i.first, expected++);
  }
}

// Test that we can convert TSC measurements back to wall clock time.
TEST(ProfilerTest, clock_converter) {
  const int n = 10001;
  c10::ApproximateClockToUnixTimeConverter converter;
  std::vector<
      c10::ApproximateClockToUnixTimeConverter::UnixAndApproximateTimePair>
      pairs;
  for (const auto i : c10::irange(n)) {
    pairs.push_back(c10::ApproximateClockToUnixTimeConverter::measurePair());
  }
  auto count_to_ns = converter.makeConverter();
  std::vector<int64_t> deltas;
  for (const auto& i : pairs) {
    deltas.push_back(i.t_ - count_to_ns(i.approx_t_));
  }
  std::sort(deltas.begin(), deltas.end());

  // In general it's not a good idea to put clocks in unit tests as it leads
  // to flakiness. We mitigate this by:
  //   1) Testing the clock itself. While the time to complete a task may
  //      vary, two clocks measuring the same time should be much more
  //      consistent.
  //   2) Only testing the interquartile range. Context switches between
  //      calls to the two timers do occur and can result in hundreds of
  //      nanoseconds of noise, but such switches are only a few percent
  //      of cases.
  //   3) We're willing to accept a somewhat large bias which can emerge from
  //      differences in the cost of calling each clock.
  EXPECT_LT(std::abs(deltas[n / 2]), 200);
  EXPECT_LT(deltas[n * 3 / 4] - deltas[n / 4], 50);
}

TEST(ProfilerTest, soft_assert) {
  EXPECT_TRUE(SOFT_ASSERT(true));
  torch::profiler::impl::setSoftAssertRaises(true);
  EXPECT_ANY_THROW(SOFT_ASSERT(false));
  torch::profiler::impl::setSoftAssertRaises(false);
  EXPECT_NO_THROW(SOFT_ASSERT(false));
  // Reset soft assert behavior to default
  torch::profiler::impl::setSoftAssertRaises(c10::nullopt);
  EXPECT_NO_THROW(SOFT_ASSERT(false));
}
