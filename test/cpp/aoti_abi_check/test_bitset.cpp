#include <gtest/gtest.h>

#include <torch/headeronly/util/Bitset.h>

#include <vector>

TEST(TestBitset, TestBitset) {
  torch::headeronly::utils::bitset b;
  EXPECT_TRUE(b.is_entirely_unset());
  b.set(3);
  b.set(5);
  EXPECT_TRUE(b.get(3));
  EXPECT_FALSE(b.get(4));

  std::vector<size_t> set_bits;
  b.for_each_set_bit([&](size_t i) { set_bits.push_back(i); });
  EXPECT_EQ(set_bits.size(), 2u);

  // c10 alias + comparison operators
  c10::utils::bitset c;
  c.set(3);
  c.set(5);
  EXPECT_TRUE(b == c);
  c.set(7);
  EXPECT_TRUE(b != c);
}
