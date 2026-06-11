#include <gtest/gtest.h>

#include <torch/headeronly/util/sparse_bitset.h>

TEST(TestSparseBitset, TestSparseBitset) {
  torch::headeronly::SparseBitVector<> bv;
  bv.set(5);
  bv.set(100);
  EXPECT_TRUE(bv.test(5));
  EXPECT_FALSE(bv.test(6));
  EXPECT_TRUE(bv.test(100));

  // exercise the element type name as a standalone token
  using Elt = torch::headeronly::SparseBitVectorElement<>;
  (void)sizeof(Elt);

  // c10 alias
  c10::SparseBitVector<> bv2;
  bv2.set(1);
  EXPECT_TRUE(bv2.test(1));
}
