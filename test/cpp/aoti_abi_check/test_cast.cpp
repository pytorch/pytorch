#include <gtest/gtest.h>

#include <torch/headeronly/util/TypeCast.h>
#include <torch/headeronly/util/bit_cast.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
namespace torch {
namespace aot_inductor {

TEST(TestCast, TestConvert) {
  using torch::headeronly::BFloat16;
  using torch::headeronly::convert;
  using torch::headeronly::Half;

  BFloat16 a = 3.0f;
  Half b = 3.0f;

  EXPECT_EQ(convert<Half>(a), b);
  EXPECT_EQ(a, convert<BFloat16>(b));
}

TEST(TestCast, TestConvertRoundsDoubleOnce) {
  using torch::headeronly::BFloat16;
  using torch::headeronly::convert;
  using torch::headeronly::Half;

  // Just above the midpoint between 1 and the next value. Rounding to float
  // first would land exactly on the midpoint, which then ties to even (1.0).
  // BFloat16 also covers aarch64, where Half converts from double natively.
  EXPECT_EQ(
      static_cast<float>(convert<Half>(1.0 + 0x1p-11 + 0x1p-40)),
      1.0f + 0x1p-10f);
  EXPECT_EQ(
      static_cast<float>(convert<BFloat16>(1.0 + 0x1p-8 + 0x1p-40)),
      1.0f + 0x1p-7f);
  const auto z =
      convert<torch::headeronly::complex<Half>>(1.0 + 0x1p-11 + 0x1p-40);
  EXPECT_EQ(static_cast<float>(z.real()), 1.0f + 0x1p-10f);
  EXPECT_EQ(static_cast<float>(z.imag()), 0.0f);
}

TEST(TestCast, TestRoundsThroughFloat) {
  using namespace torch::headeronly;
  using c10::detail::rounds_through_float;
  static_assert(
      rounds_through_float<Half> && rounds_through_float<BFloat16> &&
      rounds_through_float<Float8_e4m3fn> &&
      rounds_through_float<Float8_e4m3fnuz> &&
      rounds_through_float<Float8_e5m2> &&
      rounds_through_float<Float8_e5m2fnuz> &&
      rounds_through_float<Float8_e5m3fnu> &&
      rounds_through_float<Float8_e8m0fnu>);
  static_assert(
      !rounds_through_float<float> && !rounds_through_float<double> &&
      !rounds_through_float<int8_t> && !rounds_through_float<bool>);
}

TEST(TestCast, TestCheckedConvert) {
  using torch::headeronly::checked_convert;

  EXPECT_EQ(checked_convert<int8_t>(100, "int8_t"), 100);
  EXPECT_THROW(checked_convert<int8_t>(200, "int8_t"), std::runtime_error);
}

TEST(TestCast, TestConvertFloatToInt) {
  using torch::headeronly::convert;
  // Exercises unchecked_cast_to_int, which convert() routes through when
  // going from a non-integral source to an integral destination.
  EXPECT_EQ(convert<int32_t>(3.7), 3);
  EXPECT_EQ(convert<uint8_t>(200.0), 200);
}

TEST(TestCast, TestUnsafeWrappingConvert) {
  using torch::headeronly::unsafe_wrapping_convert;

  EXPECT_EQ(unsafe_wrapping_convert<uint8_t>(-1, "uint8_t"), 255);
}

TEST(TestCast, TestReportOverflow) {
  using torch::headeronly::report_overflow;

  EXPECT_THROW(report_overflow("int8_t"), std::runtime_error);
}

TEST(TestCast, TestBitcast) {
  using torch::headeronly::BFloat16;
  using torch::headeronly::bit_cast;
  using torch::headeronly::Half;

  BFloat16 a = 3.0f;
  Half b = 3.0f;

  EXPECT_EQ(bit_cast<BFloat16>(bit_cast<Half>(a)), a);
  EXPECT_EQ(bit_cast<Half>(bit_cast<BFloat16>(b)), b);
}

} // namespace aot_inductor
} // namespace torch
