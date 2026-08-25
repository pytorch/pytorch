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

TEST(TestCast, TestCheckedConvert) {
  using torch::headeronly::checked_convert;

  EXPECT_EQ(checked_convert<int8_t>(100, "int8_t"), 100);
  EXPECT_THROW(checked_convert<int8_t>(200, "int8_t"), std::runtime_error);
}

TEST(TestCast, TestUnsafeWrappingConvert) {
  using torch::headeronly::unsafe_wrapping_convert;

  // Unlike checked_convert, unsafe_wrapping_convert permits two's-complement
  // wraparound when converting a negative signed value to an unsigned type.
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
