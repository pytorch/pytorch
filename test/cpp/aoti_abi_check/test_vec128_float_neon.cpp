#include <gtest/gtest.h>

#if defined(__aarch64__)

#include <torch/headeronly/cpu/vec/vec128/vec128_float_neon.h>

TEST(TestVec128FloatNeon, TestVectorizedFloatAdd) {
  using Vec = at::vec::Vectorized<float>;
  float data[Vec::size()] = {1.0f, 2.0f, 3.0f, 4.0f};
  Vec a = Vec::loadu(data);
  Vec b = Vec(1.0f);
  Vec c = a + b;
  float result[Vec::size()];
  c.store(result);
  EXPECT_FLOAT_EQ(result[0], 2.0f);
  EXPECT_FLOAT_EQ(result[1], 3.0f);
  EXPECT_FLOAT_EQ(result[2], 4.0f);
  EXPECT_FLOAT_EQ(result[3], 5.0f);
}

TEST(TestVec128FloatNeon, TestAtVecNamespaceAlias) {
  using Vec = at::vec::Vectorized<float>;
  float data[Vec::size()] = {0.0f, 1.0f, 2.0f, 3.0f};
  Vec v = Vec::loadu(data);
  Vec r = v.abs();
  float result[Vec::size()];
  r.store(result);
  EXPECT_FLOAT_EQ(result[0], 0.0f);
  EXPECT_FLOAT_EQ(result[1], 1.0f);
  EXPECT_FLOAT_EQ(result[2], 2.0f);
  EXPECT_FLOAT_EQ(result[3], 3.0f);
}

#endif // defined(__aarch64__)
