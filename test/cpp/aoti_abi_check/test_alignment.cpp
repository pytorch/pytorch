#include <gtest/gtest.h>

#include <torch/headeronly/core/alignment.h>

TEST(TestAlignment, TestAlignment) {
  EXPECT_GT(torch::headeronly::gAlignment, 0u);
  EXPECT_EQ(torch::headeronly::gPagesize, 4096u);
  EXPECT_GT(torch::headeronly::gAlloc_threshold_thp, 0u);
  EXPECT_GT(torch::headeronly::hardware_destructive_interference_size, 0u);

  // Backward-compatible c10 aliases resolve to the same values.
  EXPECT_EQ(c10::gAlignment, torch::headeronly::gAlignment);
  EXPECT_EQ(c10::gPagesize, torch::headeronly::gPagesize);
}
