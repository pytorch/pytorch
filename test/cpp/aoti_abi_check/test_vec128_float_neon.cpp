#include <gtest/gtest.h>

#include <torch/headeronly/cpu/vec/vec128/vec128_float_neon.h>

#include <array>


TEST(TestVec128FloatNeon, TestBlend) {
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
  using Vec = torch::headeronly::vec::Vectorized<float>;
  const Vec zeros(0.0f);
  const Vec ones(1.0f);
  const Vec actual = Vec::blend<0b0101>(zeros, ones);

  std::array<float, Vec::size()> output{};
  actual.store(output.data());

  EXPECT_FLOAT_EQ(output[0], 1.0f);
  EXPECT_FLOAT_EQ(output[1], 0.0f);
  EXPECT_FLOAT_EQ(output[2], 1.0f);
  EXPECT_FLOAT_EQ(output[3], 0.0f);
#else
  GTEST_SKIP() << "Requires AArch64 NEON";
#endif
}

TEST(TestVec128FloatNeon, TestBlendv) {
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
  using Vec = torch::headeronly::vec::Vectorized<float>;
  const Vec zeros(0.0f);
  const Vec ones(1.0f);
  const Vec mask = zeros < ones;
  const Vec actual = Vec::blendv(zeros, ones, mask);

  std::array<float, Vec::size()> output{};
  actual.store(output.data());
  for (float value : output) {
    EXPECT_FLOAT_EQ(value, 1.0f);
  }
#else
  GTEST_SKIP() << "Requires AArch64 NEON";
#endif
}

TEST(TestVec128FloatNeon, TestExpU20) {
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
  using Vec = torch::headeronly::vec::Vectorized<float>;
  const Vec actual = Vec(0.0f).exp_u20();

  std::array<float, Vec::size()> output{};
  actual.store(output.data());
  for (float value : output) {
    EXPECT_FLOAT_EQ(value, 1.0f);
  }
#else
  GTEST_SKIP() << "Requires AArch64 NEON";
#endif
}

TEST(TestVec128FloatNeon, TestFexpU20) {
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
  using Vec = torch::headeronly::vec::Vectorized<float>;
  const Vec actual = Vec(0.0f).fexp_u20();

  std::array<float, Vec::size()> output{};
  actual.store(output.data());
  for (float value : output) {
    EXPECT_FLOAT_EQ(value, 1.0f);
  }
#else
  GTEST_SKIP() << "Requires AArch64 NEON";
#endif
}

TEST(TestVec128FloatNeon, TestErf) {
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
  using Vec = torch::headeronly::vec::Vectorized<float>;
  const Vec actual = Vec(0.0f).erf();

  std::array<float, Vec::size()> output{};
  actual.store(output.data());
  for (float value : output) {
    EXPECT_FLOAT_EQ(value, 0.0f);
  }
#else
  GTEST_SKIP() << "Requires AArch64 NEON";
#endif
}
