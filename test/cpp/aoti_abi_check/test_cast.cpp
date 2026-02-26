#include <gtest/gtest.h>

#include <c10/util/TypeCast.h>
#include <c10/util/bit_cast.h>
namespace torch {
namespace aot_inductor {

TEST(TestCast, TestConvert) {
  c10::BFloat16 a = 3.0f;
  c10::Half b = 3.0f;

  EXPECT_EQ(c10::convert<c10::Half>(a), b);
  EXPECT_EQ(a, c10::convert<c10::BFloat16>(b));
}

TEST(TestCast, TestBitcast) {
  c10::BFloat16 a = 3.0f;
  c10::Half b = 3.0f;

  EXPECT_EQ(c10::bit_cast<c10::BFloat16>(c10::bit_cast<c10::Half>(a)), a);
  EXPECT_EQ(c10::bit_cast<c10::Half>(c10::bit_cast<c10::BFloat16>(b)), b);
}

} // namespace aot_inductor
} // namespace torch
