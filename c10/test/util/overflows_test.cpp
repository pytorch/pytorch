#include <c10/util/overflows.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>

// Regression tests for the float->int range check in c10::overflows. The bug:
// limit::max() for wide integer types is not exactly representable in floating
// point (e.g. int64 max = 2^63-1 rounds up to 2^63), so `f > max()` used to let
// double(2^63) slip through, which then became INT64_MIN via static_cast. The
// fix compares against the exactly-representable bound max()+1 == 2^digits.

namespace c10 {
namespace {

TEST(overflowsTest, FloatToWideIntBoundaryHigh) {
  // The smoking gun: 2^63 is exactly representable in double and is
  // INT64_MAX+1.
  EXPECT_TRUE(overflows<int64_t>(9223372036854775808.0)); // 2^63
  // The largest double strictly below 2^63 is in range.
  EXPECT_FALSE(overflows<int64_t>(9223372036854774784.0)); // 2^63 - 1024
  EXPECT_TRUE(overflows<uint64_t>(18446744073709551616.0)); // 2^64
  EXPECT_FALSE(overflows<uint64_t>(9223372036854775808.0)); // 2^63, fits u64
}

TEST(overflowsTest, FloatToInt32Boundary) {
  EXPECT_TRUE(overflows<int32_t>(2147483648.0)); // 2^31 == INT32_MAX+1
  EXPECT_FALSE(overflows<int32_t>(2147483647.0)); // INT32_MAX, exact in double
  EXPECT_TRUE(overflows<uint32_t>(4294967296.0)); // 2^32
  EXPECT_FALSE(
      overflows<uint32_t>(4294967295.0)); // UINT32_MAX, exact in double
  // float has fewer mantissa bits: 2^31 as float is out of int32 range.
  EXPECT_TRUE(overflows<int32_t>(2147483648.0f));
}

TEST(overflowsTest, FloatToNarrowIntBoundary) {
  EXPECT_TRUE(overflows<int8_t>(128.0));
  EXPECT_FALSE(overflows<int8_t>(127.0));
  EXPECT_FALSE(overflows<int8_t>(-128.0));
  EXPECT_TRUE(overflows<uint8_t>(256.0));
  EXPECT_FALSE(overflows<uint8_t>(255.0));
  EXPECT_TRUE(overflows<uint8_t>(-1.0)); // negative -> unsigned
}

TEST(overflowsTest, FloatNanAndInf) {
  // NaN is not representable in any integer type.
  EXPECT_TRUE(overflows<int32_t>(std::nan("")));
  EXPECT_TRUE(overflows<int64_t>(std::nan("")));
  // Inf to a float target with infinity is not an overflow; to int it is.
  EXPECT_FALSE(overflows<float>(std::numeric_limits<double>::infinity()));
  EXPECT_TRUE(overflows<int64_t>(std::numeric_limits<double>::infinity()));
}

// static_cast after a passing check must reproduce the value (no silent wrap).
TEST(overflowsTest, InRangeFloatStaticCastIsExact) {
  const double v =
      9223372036854774784.0; // 2^63 - 1024, representable, in range
  ASSERT_FALSE(overflows<int64_t>(v));
  EXPECT_EQ(static_cast<int64_t>(v), int64_t{9223372036854774784LL});
}

// The core invariant the fix guarantees: whenever overflows() reports a value
// as in range, the subsequent static_cast must not silently wrap. A wrap would
// make the round-tripped value differ from the input by >= 1 (truncation loses
// < 1).
template <typename To>
void expectNoSilentWrap(double f) {
  if (!overflows<To>(f)) {
    const double back = static_cast<double>(static_cast<To>(f));
    EXPECT_LT(std::abs(back - f), 1.0)
        << "silent wrap converting " << f << " to a "
        << std::numeric_limits<To>::digits << "-digit integer";
  }
}

template <typename To>
void sweepNoSilentWrap() {
  constexpr double probes[] = {
      0,
      127,
      128,
      -128,
      -129,
      255,
      256,
      32767,
      32768,
      65535,
      65536,
      -32769,
      2147483647.0,
      2147483648.0,
      4294967295.0,
      4294967296.0,
      -2147483649.0,
      9223372036854774784.0,
      9223372036854775808.0,
      18446744073709551616.0,
      -9223372036854775808.0,
      1e30,
      -1e30};
  for (double p : probes) {
    expectNoSilentWrap<To>(p);
    expectNoSilentWrap<To>(p + 0.5);
    expectNoSilentWrap<To>(p - 0.5);
  }
  for (int i = -100000; i <= 100000; i += 7) {
    expectNoSilentWrap<To>(i * 1.0);
    expectNoSilentWrap<To>(i + 0.25);
  }
}

TEST(overflowsTest, NoSilentWrapSweep) {
  sweepNoSilentWrap<int8_t>();
  sweepNoSilentWrap<uint8_t>();
  sweepNoSilentWrap<int16_t>();
  sweepNoSilentWrap<uint16_t>();
  sweepNoSilentWrap<int32_t>();
  sweepNoSilentWrap<uint32_t>();
  sweepNoSilentWrap<int64_t>();
  sweepNoSilentWrap<uint64_t>();
}

} // namespace
} // namespace c10
