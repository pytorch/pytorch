#include <gtest/gtest.h>

#include <torch/headeronly/util/copysign.h>

TEST(TestCopysign, TestCopysign) {
  EXPECT_EQ(torch::headeronly::copysign(3.0, -1.0), -3.0);
  EXPECT_EQ(c10::copysign(3.0f, 1.0f), 3.0f);

  using torch::headeronly::BFloat16;
  using torch::headeronly::Half;
  Half h = torch::headeronly::copysign(Half(2.0f), Half(-1.0f));
  EXPECT_LT(static_cast<float>(h), 0.0f);
  BFloat16 b = c10::copysign(BFloat16(2.0f), BFloat16(-1.0f));
  EXPECT_LT(static_cast<float>(b), 0.0f);
}
