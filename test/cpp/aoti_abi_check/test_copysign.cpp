#include <gtest/gtest.h>

#include <torch/headeronly/util/copysign.h>

namespace torch {
namespace aot_inductor {

TEST(TestCopysign, TestFloatingPoint) {
  using torch::headeronly::copysign;
  EXPECT_EQ(copysign(3.0, -1.0), -3.0);
  EXPECT_EQ(copysign(-3.0f, 1.0f), 3.0f);
}

TEST(TestCopysign, TestHalfAndBFloat16) {
  using torch::headeronly::BFloat16;
  using torch::headeronly::copysign;
  using torch::headeronly::Half;

  Half h_pos(3.0f);
  Half h_neg(-1.0f);
  EXPECT_EQ(copysign(h_pos, h_neg), Half(-3.0f));
  EXPECT_EQ(copysign(h_neg, h_pos), Half(1.0f));

  BFloat16 b_pos(3.0f);
  BFloat16 b_neg(-1.0f);
  EXPECT_EQ(copysign(b_pos, b_neg), BFloat16(-3.0f));
  EXPECT_EQ(copysign(b_neg, b_pos), BFloat16(1.0f));
}

} // namespace aot_inductor
} // namespace torch
