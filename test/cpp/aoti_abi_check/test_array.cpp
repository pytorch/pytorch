#include <gtest/gtest.h>

#include <torch/headeronly/util/Array.h>

TEST(TestArray, TestArrayOf) {
  constexpr auto a = torch::headeronly::array_of<int>(1, 2, 3);
  static_assert(a.size() == 3);
  EXPECT_EQ(a[0], 1);
  EXPECT_EQ(a[2], 3);

  // c10 alias
  auto b = c10::array_of<double>(1.0, 2.0);
  EXPECT_EQ(b.size(), 2u);
}
