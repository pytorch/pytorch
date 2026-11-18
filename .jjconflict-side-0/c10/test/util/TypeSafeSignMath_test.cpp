#include <c10/util/TypeSafeSignMath.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <typeinfo>
#include <utility>

// These tests pin the invariant that c10::safe_conv relies on: for
// integer->integer, the TypeSafeSignMath range check
//   !less_than_lowest<To>(f) && !greater_than_max<To>(f)
// is exactly equivalent to std::in_range<To>(f) (the C++20 ground truth). This
// guards against regressions in the hand-written sign-aware comparisons that
// safe_conv uses in place of std::in_range (see c10/util/safe_conv.h).

namespace c10 {
namespace {

template <typename To, typename From>
bool tsm_representable(From f) {
  return !c10::less_than_lowest<To>(f) && !c10::greater_than_max<To>(f);
}

template <typename To, typename From>
void checkOne(From f) {
  EXPECT_EQ(tsm_representable<To>(f), std::in_range<To>(f))
      << "To=" << typeid(To).name() << " From=" << typeid(From).name()
      << " f=" << static_cast<long long>(f);
}

// Every target integer type, for a single source value.
template <typename From>
void checkAllTargets(From f) {
  checkOne<int8_t>(f);
  checkOne<uint8_t>(f);
  checkOne<int16_t>(f);
  checkOne<uint16_t>(f);
  checkOne<int32_t>(f);
  checkOne<uint32_t>(f);
  checkOne<int64_t>(f);
  checkOne<uint64_t>(f);
}

// Boundary seeds spanning every integer type's limits (+/-1 around them). Cast
// to the source type (integer static_cast is always well-defined) so each
// source type is exercised right where representability flips.
template <typename From>
void checkBoundarySeeds() {
  constexpr int64_t s[] = {
      0,
      1,
      2,
      -1,
      -2,
      126,
      127,
      128,
      254,
      255,
      256,
      -127,
      -128,
      -129,
      32766,
      32767,
      32768,
      65535,
      65536,
      -32768,
      -32769,
      2147483646LL,
      2147483647LL,
      2147483648LL,
      4294967295LL,
      4294967296LL,
      -2147483648LL,
      -2147483649LL,
      9223372036854775806LL,
      9223372036854775807LL,
      std::numeric_limits<int64_t>::min()};
  for (int64_t v : s) {
    checkAllTargets<From>(static_cast<From>(v));
  }
  constexpr uint64_t u[] = {
      9223372036854775808ULL, 18446744073709551614ULL, 18446744073709551615ULL};
  for (uint64_t v : u) {
    checkAllTargets<From>(static_cast<From>(v));
  }
  checkAllTargets<From>(std::numeric_limits<From>::min());
  checkAllTargets<From>(std::numeric_limits<From>::max());
}

TEST(TypeSafeSignMathTest, ExhaustiveEightAndSixteenBitSources) {
  for (int v = 0; v <= 255; ++v) {
    checkAllTargets<uint8_t>(static_cast<uint8_t>(v));
  }
  for (int v = -128; v <= 127; ++v) {
    checkAllTargets<int8_t>(static_cast<int8_t>(v));
  }
  for (int v = 0; v <= 65535; ++v) {
    checkAllTargets<uint16_t>(static_cast<uint16_t>(v));
  }
  for (int v = -32768; v <= 32767; ++v) {
    checkAllTargets<int16_t>(static_cast<int16_t>(v));
  }
}

TEST(TypeSafeSignMathTest, BoundarySeedsWideSources) {
  checkBoundarySeeds<int32_t>();
  checkBoundarySeeds<uint32_t>();
  checkBoundarySeeds<int64_t>();
  checkBoundarySeeds<uint64_t>();
}

// The behaviors safe_conv depends on, pinned explicitly so the critical rows
// are protected even if the oracle above ever changes.
TEST(TypeSafeSignMathTest, CriticalRows) {
  // negative -> unsigned is out of range (no two's-complement wrap)
  EXPECT_FALSE(tsm_representable<uint32_t>(int64_t{-1}));
  EXPECT_FALSE(tsm_representable<uint8_t>(int32_t{-5}));
  EXPECT_FALSE(tsm_representable<uint64_t>(int64_t{-1}));
  // exact-fit boundaries
  EXPECT_TRUE(tsm_representable<int32_t>(int64_t{2147483647}));
  EXPECT_FALSE(tsm_representable<int32_t>(int64_t{2147483648}));
  EXPECT_TRUE(tsm_representable<uint32_t>(int64_t{4294967295}));
  EXPECT_FALSE(tsm_representable<uint32_t>(int64_t{4294967296}));
  EXPECT_TRUE(tsm_representable<int64_t>(uint64_t{9223372036854775807ULL}));
  EXPECT_FALSE(tsm_representable<int64_t>(uint64_t{9223372036854775808ULL}));
  EXPECT_TRUE(tsm_representable<uint64_t>(uint64_t{18446744073709551615ULL}));
  // same-type is always representable
  EXPECT_TRUE(
      tsm_representable<uint64_t>(std::numeric_limits<uint64_t>::max()));
  EXPECT_TRUE(tsm_representable<int8_t>(int8_t{-128}));
}

} // namespace
} // namespace c10
