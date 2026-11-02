#include <c10/metal/double_float.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using c10::metal::df32;

namespace {

// The value a df32 stands for, evaluated exactly: hi and lo are float32 and
// do not overlap, so their sum is representable in double.
double value(df32 v) {
  return static_cast<double>(v.hi) + static_cast<double>(v.lo);
}

constexpr double kOperands[] = {1.0, 0.1, 3.7e5, -2.5e-7, 1234567.75};
constexpr double kDivisors[] = {3.0, 0.001, -7.3e3, 6.25e-4, 1.0 / 3.0};

} // namespace

// A double rounds into a df32 with far more of it retained than a float32
// keeps, though not all of it: 48 bits of significand against 53.
TEST(DoubleFloatTest, SplitsADouble) {
  for (double x : kOperands) {
    EXPECT_NEAR(value(df32(x)), x, std::abs(x) * 1e-15);
  }
}

// Non-finite inputs have no residual to carry, and must not leave a NaN in lo.
TEST(DoubleFloatTest, SplitOfNonFiniteHasNoResidual) {
  const df32 inf(std::numeric_limits<double>::infinity());
  EXPECT_TRUE(std::isinf(inf.hi));
  EXPECT_EQ(inf.lo, 0.0f);

  const df32 too_large(1e300);
  EXPECT_TRUE(std::isinf(too_large.hi));
  EXPECT_EQ(too_large.lo, 0.0f);
}

TEST(DoubleFloatTest, ArithmeticTracksDouble) {
  for (double x : kOperands) {
    for (double y : kDivisors) {
      const df32 a(x), b(y);
      const double xa = value(a), yb = value(b);

      EXPECT_NEAR(
          value(c10::metal::add(a, b)), xa + yb, std::abs(xa + yb) * 1e-13);
      EXPECT_NEAR(
          value(c10::metal::sub(a, b)), xa - yb, std::abs(xa - yb) * 1e-13);
      EXPECT_NEAR(
          value(c10::metal::mul(a, b)), xa * yb, std::abs(xa * yb) * 1e-13);
      // div refines a float32 quotient rather than computing an exact one, so
      // it is the loosest of the four.
      EXPECT_NEAR(
          value(c10::metal::div(a, b)), xa / yb, std::abs(xa / yb) * 1e-12);
    }
  }
}

// two_sum and two_prod are error-free transformations: the pair they return is
// the unrounded result, not an approximation of it.
TEST(DoubleFloatTest, TransformationsAreExact) {
  const float a = 1.0000001f, b = 3.9999998f;
  EXPECT_EQ(
      value(c10::metal::two_prod(a, b)),
      static_cast<double>(a) * static_cast<double>(b));
  EXPECT_EQ(
      value(c10::metal::two_sum(a, b)),
      static_cast<double>(a) + static_cast<double>(b));

  // Catastrophic cancellation: the difference is exact even though the leading
  // word alone loses every significant bit of it.
  const float big = 1 << 24, small = 1.0f;
  EXPECT_EQ(
      value(c10::metal::two_sum(big, -small)), static_cast<double>(big) - 1.0);
}

// Indices are converted exactly well past the 2^24 where float32 stops
// representing consecutive integers.
TEST(DoubleFloatTest, ConvertsLargeIntegersExactly) {
  for (long i :
       {1L << 24, (1L << 24) + 1, (1L << 40) + 12345L, -((1L << 31) + 7L)}) {
    EXPECT_EQ(value(df32(i)), static_cast<double>(i));
  }
}
