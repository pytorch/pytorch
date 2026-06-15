#include <gtest/gtest.h>

#include <torch/headeronly/util/SmallBuffer.h>

TEST(TestSmallBuffer, TestSmallBuffer) {
  // inline storage path (size <= N)
  torch::headeronly::SmallBuffer<int, 4> small(3);
  for (size_t i = 0; i < small.size(); ++i) {
    small[i] = static_cast<int>(i) * 10;
  }
  EXPECT_EQ(small.size(), 3u);
  EXPECT_EQ(small[2], 20);
  int sum = 0;
  for (int v : small) {
    sum += v;
  }
  EXPECT_EQ(sum, 30);

  // heap path (size > N), via c10 alias
  c10::SmallBuffer<int, 2> big(5);
  EXPECT_EQ(big.size(), 5u);
  big[4] = 7;
  EXPECT_EQ(big[4], 7);
}
