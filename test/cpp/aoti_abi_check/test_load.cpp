#include <gtest/gtest.h>

#include <torch/headeronly/util/Load.h>

TEST(TestLoad, TestLoad) {
  int x = 42;
  EXPECT_EQ(torch::headeronly::load<int>(&x), 42);

  // Loading a bool from an invalid (non 0/1) byte yields true, not UB.
  unsigned char b = 2;
  bool loaded = c10::load<bool>(static_cast<const void*>(&b));
  EXPECT_TRUE(loaded);

  float f = 1.5f;
  EXPECT_EQ(c10::load(&f), 1.5f);
}
