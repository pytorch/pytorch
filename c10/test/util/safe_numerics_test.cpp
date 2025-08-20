#include <c10/util/safe_numerics.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <limits>

namespace {
template <typename T>
std::optional<T> mul_overflows(T a, T b) {
  T result{0};
  if (c10::mul_overflows(a, b, &result))
    return std::nullopt;
  return result;
}
} // namespace

TEST(MulOverflowsTest, Int64BasicOperations) {
  ASSERT_EQ(mul_overflows<int64_t>(2, 3), 6);
  ASSERT_EQ(mul_overflows<int64_t>(-2, 3), -6);
  ASSERT_EQ(mul_overflows<int64_t>(2, -3), -6);
  ASSERT_EQ(mul_overflows<int64_t>(-2, -3), 6);
  ASSERT_EQ(mul_overflows<int64_t>(0, 5), 0);
  ASSERT_EQ(mul_overflows<int64_t>(5, 0), 0);

  // One cases
  ASSERT_EQ(mul_overflows<int64_t>(1, 1), 1);
  ASSERT_EQ(mul_overflows<int64_t>(1, -1), -1);
  ASSERT_EQ(mul_overflows<int64_t>(-1, 1), -1);
  ASSERT_EQ(mul_overflows<int64_t>(-1, -1), 1);
}

TEST(MulOverflowsTest, Int64OverflowCases) {
  const int64_t max_val = std::numeric_limits<int64_t>::max();
  const int64_t min_val = std::numeric_limits<int64_t>::min();

  ASSERT_EQ(mul_overflows<int64_t>(max_val, 2), std::nullopt);
  ASSERT_EQ(mul_overflows<int64_t>(2, max_val), std::nullopt);
  ASSERT_EQ(mul_overflows<int64_t>(max_val, max_val), std::nullopt);

  ASSERT_EQ(mul_overflows<int64_t>(min_val, 2), std::nullopt);
  ASSERT_EQ(mul_overflows<int64_t>(2, min_val), std::nullopt);
  ASSERT_EQ(mul_overflows<int64_t>(min_val, min_val), std::nullopt);
}

TEST(MulOverflowsTest, Int64LargeNumbersNoOverflow) {
  const int64_t max_val = std::numeric_limits<int64_t>::max();
  const int64_t min_val = std::numeric_limits<int64_t>::min();

  ASSERT_EQ(mul_overflows<int64_t>(1L << 30L, 1L << 30L), 1L << 60);
  ASSERT_EQ(mul_overflows<int64_t>(1L << 32L, 1L << 32L), std::nullopt);
  ASSERT_EQ(mul_overflows<int64_t>(-(1L << 30L), 1L << 30L), -(1L << 60L));
  ASSERT_EQ(mul_overflows<int64_t>(1L << 32L, -(1L << 32L)), std::nullopt);

  // max_val is odd, so max_val/2 * 2 = max_val - 1
  ASSERT_EQ(mul_overflows<int64_t>(max_val / 2, 2), max_val - 1);
  ASSERT_EQ(mul_overflows<int64_t>(min_val / 2, 2), min_val);
}

TEST(MulOverflowsTest, Int64BoundaryValues) {
  const int64_t max_val = std::numeric_limits<int64_t>::max();
  const int64_t min_val = std::numeric_limits<int64_t>::min();

  ASSERT_EQ(mul_overflows<int64_t>(1, max_val), max_val);
  ASSERT_EQ(mul_overflows<int64_t>(max_val, 1), max_val);
  ASSERT_EQ(mul_overflows<int64_t>(1, min_val), min_val);
  ASSERT_EQ(mul_overflows<int64_t>(min_val, 1), min_val);
  ASSERT_EQ(mul_overflows<int64_t>(-1, max_val), -max_val);
  ASSERT_EQ(mul_overflows<int64_t>(max_val, -1), -max_val);

  // result would be -min_val which is > max_val
  ASSERT_EQ(mul_overflows<int64_t>(min_val, -1L), std::nullopt);
  ASSERT_EQ(mul_overflows<int64_t>(-1L, min_val), std::nullopt);
}

TEST(MulOverflowsTest, Uint64BasicOperations) {
  ASSERT_EQ(mul_overflows<uint64_t>(2u, 3u), 6u);
  ASSERT_EQ(mul_overflows<uint64_t>(0u, 5u), 0u);
  ASSERT_EQ(mul_overflows<uint64_t>(5u, 0u), 0u);
  ASSERT_EQ(mul_overflows<uint64_t>(100u, 200u), 20000u);
}

TEST(MulOverflowsTest, Uint64OverflowCases) {
  const uint64_t max_val = std::numeric_limits<uint64_t>::max();

  ASSERT_EQ(mul_overflows<uint64_t>(max_val, 2u), std::nullopt);
  ASSERT_EQ(mul_overflows<uint64_t>(2ull, max_val), std::nullopt);
  ASSERT_EQ(mul_overflows<uint64_t>(1ull << 32, 1ull << 32), std::nullopt);
  ASSERT_EQ(mul_overflows<uint64_t>(max_val, max_val), std::nullopt);

  uint64_t floor_sqrt_max = 4294967295ull;
  ASSERT_EQ(
      mul_overflows<uint64_t>(floor_sqrt_max + 1ull, floor_sqrt_max + 1ull),
      std::nullopt);
}

TEST(MulOverflowsTest, Uint64LargeNumbersNoOverflow) {
  const uint64_t max_val = std::numeric_limits<uint64_t>::max();

  // max_val is odd, so max_val/2 * 2 = max_val - 1
  ASSERT_EQ(mul_overflows<uint64_t>(max_val / 2u, 2u), max_val - 1u);
  ASSERT_EQ(mul_overflows<uint64_t>(1ull << 31, 1ull << 31), 1ull << 62);

  uint64_t floor_sqrt_max = 4294967295ull;
  ASSERT_EQ(
      mul_overflows<uint64_t>(floor_sqrt_max, floor_sqrt_max),
      18446744065119617025ull);
}

TEST(MulOverflowsTest, Uint64BoundaryValues) {
  const uint64_t max_val = std::numeric_limits<uint64_t>::max();

  ASSERT_EQ(mul_overflows<uint64_t>(1u, max_val), max_val);
  ASSERT_EQ(mul_overflows<uint64_t>(max_val, 1u), max_val);
  ASSERT_EQ(mul_overflows<uint64_t>(0u, max_val), 0u);
  ASSERT_EQ(mul_overflows<uint64_t>(max_val, 0u), 0u);
}
