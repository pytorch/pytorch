#include <gtest/gtest.h>

#include <torch/headeronly/util/Unroll.h>

TEST(TestUnroll, TestForcedUnroll) {
  int sum = 0;
  torch::headeronly::ForcedUnroll<4>{}(
      [&](auto i) { sum += static_cast<int>(decltype(i)::value); });
  EXPECT_EQ(sum, 0 + 1 + 2 + 3);

  int product = 1;
  c10::ForcedUnroll<3>{}(
      [&](auto i) { product *= (static_cast<int>(decltype(i)::value) + 1); });
  EXPECT_EQ(product, 1 * 2 * 3);
}
