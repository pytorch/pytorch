#include <gtest/gtest.h>
#include <c10/core/TensorLayoutPermutation.h>

using namespace c10;

TEST(TensorLayoutPermutation, Construct) {
  TensorLayoutPermutation lp1;
  TensorLayoutPermutation lp2(lp1);
  TensorLayoutPermutation lp3(lp2.get());
  TensorLayoutPermutation lp4(0xffffffffffff0123);

  ASSERT_TRUE(lp1.is_unknown());
  ASSERT_EQ(lp1, lp2);
  ASSERT_EQ(lp1.get(), lp3.get());
  ASSERT_EQ(lp4.get(), 0xffffffffffff0123);
}

TEST(TensorLayoutPermutation, Setters) {
  TensorLayoutPermutation lp;
  lp.set_by_ndim(0);
  ASSERT_TRUE(lp.is_unknown());
  lp.set_by_ndim(5);
  ASSERT_EQ(lp.get(), 0xfffffffffff01234);
  lp.reset();
  ASSERT_TRUE(lp.is_unknown());
  lp.set_by_dims({0,2,1,3,4});
  ASSERT_EQ(lp.get(), 0xfffffffffff02134);
  lp.set_by_perm(0xffffffff01234567);
  ASSERT_EQ(lp.get(), 0xffffffff01234567);
}

TEST(TensorLayoutPermutation, Permute) {
  TensorLayoutPermutation lp(0xffffffffffff0123);
  TensorLayoutPermutation lp1 = lp.permute({0,3,1,2});
  ASSERT_EQ(lp1.get(), 0xffffffffffff0312);
  lp1 = lp1.permute({0,2,3,1});
  ASSERT_EQ(lp1, lp);
  lp1.permute_({0,2,1,3});
  ASSERT_EQ(lp1.get(), 0xffffffffffff0213);
}

TEST(TensorLayoutPermutation, EqualNDimCheck) {
  TensorLayoutPermutation lp(0xffffffffffff0123);
  ASSERT_TRUE(lp.has_equal_ndim(4));
  ASSERT_FALSE(lp.has_equal_ndim(5));
  ASSERT_FALSE(lp.has_equal_ndim(0));
}

