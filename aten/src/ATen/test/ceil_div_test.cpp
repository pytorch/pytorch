#include <gtest/gtest.h>

#include <ATen/ceil_div.h>

TEST(CeilDivTest, Basic) {
  EXPECT_EQ(at::ceil_div(10, 3), 4);
  EXPECT_EQ(at::ceil_div(7, 2), 4);
  EXPECT_EQ(at::ceil_div(1, 1), 1);

  EXPECT_EQ(at::ceil_div(10L, 3), 4);
  EXPECT_EQ(at::ceil_div(7, 2L), 4);
  EXPECT_EQ(at::ceil_div(1L, 1L), 1);
}
