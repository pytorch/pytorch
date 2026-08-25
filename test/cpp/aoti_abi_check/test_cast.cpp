#include <gtest/gtest.h>

#include <c10/util/TypeCast.h>
#include <c10/util/bit_cast.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
namespace torch {
namespace aot_inductor {

TEST(TestCast, TestConvert) {
  c10::BFloat16 a = 3.0f;
  c10::Half b = 3.0f;

  EXPECT_EQ(c10::convert<c10::Half>(a), b);
  EXPECT_EQ(a, c10::convert<c10::BFloat16>(b));
}

TEST(TestCast, TestCheckedConvert) {
  EXPECT_EQ(c10::checked_convert<int8_t>(100, "int8_t"), 100);
  EXPECT_THROW(c10::checked_convert<int8_t>(200, "int8_t"), std::runtime_error);
}

TEST(TestCast, TestUnsafeWrappingConvert) {
  // Unlike checked_convert, unsafe_wrapping_convert permits two's-complement
  // wraparound when converting a negative signed value to an unsigned type.
  EXPECT_EQ(c10::unsafe_wrapping_convert<uint8_t>(-1, "uint8_t"), 255);
}

TEST(TestCast, TestReportOverflow) {
  EXPECT_THROW(c10::report_overflow("int8_t"), std::runtime_error);
}

TEST(TestCast, TestBitcast) {
  c10::BFloat16 a = 3.0f;
  c10::Half b = 3.0f;

  EXPECT_EQ(c10::bit_cast<c10::BFloat16>(c10::bit_cast<c10::Half>(a)), a);
  EXPECT_EQ(c10::bit_cast<c10::Half>(c10::bit_cast<c10::BFloat16>(b)), b);
}

} // namespace aot_inductor
} // namespace torch
