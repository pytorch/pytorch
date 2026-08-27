#include <gtest/gtest.h>

#include <torch/headeronly/cpu/vec/vec_base.h>

TEST(TestVecBase, TestGenericVectorizedAdd) {
  using Vec = at::vec::Vectorized<int>;
  int data[Vec::size()] = {};
  for (int i = 0; i < Vec::size(); ++i) {
    data[i] = i;
  }
  Vec a = Vec::loadu(data);
  Vec b = Vec::loadu(data);
  Vec c = a + b;
  int result[Vec::size()];
  c.store(result);
  for (int i = 0; i < Vec::size(); ++i) {
    EXPECT_EQ(result[i], 2 * i);
  }
}

TEST(TestVecBase, TestErfinvViaMap) {
  using Vec = at::vec::Vectorized<float>;
  float data[Vec::size()] = {0.0f, 0.5f, -0.5f, 0.0f};
  Vec v = Vec::loadu(data);
  Vec r = v.erfinv();
  float result[Vec::size()];
  r.store(result);
  EXPECT_NEAR(result[0], 0.0f, 1e-5f);
  EXPECT_NEAR(result[1], 0.476936f, 1e-4f);
  EXPECT_NEAR(result[2], -0.476936f, 1e-4f);
}

TEST(TestVecBase, TestAtVecNamespaceAlias) {
  using Vec = at::vec::Vectorized<double>;
  double data[Vec::size()] = {1.0, 2.0, 3.0, 4.0};
  Vec a = Vec::loadu(data);
  Vec b = at::vec::maximum(a, Vec(2.0));
  double result[Vec::size()];
  b.store(result);
  EXPECT_DOUBLE_EQ(result[0], 2.0);
  EXPECT_DOUBLE_EQ(result[1], 2.0);
  EXPECT_DOUBLE_EQ(result[2], 3.0);
  EXPECT_DOUBLE_EQ(result[3], 4.0);
}
