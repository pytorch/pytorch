#include <gtest/gtest.h>

#include <torch/headeronly/cpu/vec/vec128/vec128_float_neon.h>

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
namespace {

using Vec = torch::headeronly::vec::Vectorized<float>;

void ExpectFirstLaneNear(
    const Vec& actual,
    float expected,
    float tolerance = 1e-5f) {
  std::array<float, Vec::size()> output{};
  actual.store(output.data());
  EXPECT_NEAR(output[0], expected, tolerance);
}

void ExpectFirstLaneEq(const Vec& actual, float expected) {
  std::array<float, Vec::size()> output{};
  actual.store(output.data());
  EXPECT_FLOAT_EQ(output[0], expected);
}

} // namespace
#endif

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

TEST(TestVec128FloatNeon, TestMap2) {
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
  const auto add = [](float a, float b) { return a + b; };
  ExpectFirstLaneEq(Vec(1.0f).map2(Vec(2.0f), add), 3.0f);
#else
  GTEST_SKIP() << "Requires AArch64 NEON";
#endif
}

TEST(TestVec128FloatNeon, TestVexpqF32U20) {
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
  ExpectFirstLaneNear(Vec(0.0f).vexpq_f32_u20(), 1.0f);
#else
  GTEST_SKIP() << "Requires AArch64 NEON";
#endif
}

TEST(TestVec128FloatNeon, TestConstructionAndTraits) {
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
  float values[Vec::size()] = {0.5f, 0.5f, 0.5f, 0.5f};
  const float32x4_t native_values = vdupq_n_f32(0.5f);
  const Vec from_native(native_values);
  const Vec from_lanes(0.5f, 0.5f, 0.5f, 0.5f);
  const Vec from_array(values);
  const float32x4_t converted = from_native;
  (void)converted;
  EXPECT_FLOAT_EQ(from_lanes[0], 0.5f);
  EXPECT_FLOAT_EQ(from_array[0], 0.5f);

  static_assert(torch::headeronly::vec::is_vec_specialized_for<float>::value);
  static_assert(torch::headeronly::vec::is_vec_specialized_for_v<float>);
  static_assert(std::is_same_v<Vec::value_type, float>);
  static_assert(std::is_same_v<Vec::size_type, int>);
  static_assert(Vec::size() == 4);
  ExpectFirstLaneEq(Vec::loadu(values), 0.5f);
#else
  GTEST_SKIP() << "Requires AArch64 NEON";
#endif
}

TEST(TestVec128FloatNeon, TestCoreHelpers) {
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
  const Vec half(0.5f);
  const Vec one(1.0f);
  const auto square_root = [](float value) { return std::sqrt(value); };

  ExpectFirstLaneEq(Vec::arange(1.0f, 1.0f), 1.0f);
  ExpectFirstLaneEq(Vec::set(Vec(0.0f), one, 1), 1.0f);
  EXPECT_EQ(Vec(0.0f).zero_mask(), 0b1111);
  EXPECT_TRUE(Vec(std::numeric_limits<float>::quiet_NaN()).has_inf_nan());
  EXPECT_EQ(
      Vec(std::numeric_limits<float>::quiet_NaN()).isnan().zero_mask(), 0);
  ExpectFirstLaneNear(half.map(square_root), std::sqrt(0.5f));
  ExpectFirstLaneEq(Vec(-0.5f).abs(), 0.5f);
  ExpectFirstLaneNear(Vec(-0.5f).angle(), c10::pi<float>);
  ExpectFirstLaneEq(half.real(), 0.5f);
  ExpectFirstLaneEq(half.imag(), 0.0f);
  ExpectFirstLaneEq(half.conj(), 0.5f);
#else
  GTEST_SKIP() << "Requires AArch64 NEON";
#endif
}

TEST(TestVec128FloatNeon, TestComparisonHelpers) {
#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
  const Vec one(1.0f);
  const Vec two(2.0f);

  ExpectFirstLaneEq(one.eq(one), 1.0f);
  ExpectFirstLaneEq(one.ne(two), 1.0f);
  ExpectFirstLaneEq(two.gt(one), 1.0f);
  ExpectFirstLaneEq(two.ge(two), 1.0f);
  ExpectFirstLaneEq(one.lt(two), 1.0f);
  ExpectFirstLaneEq(one.le(one), 1.0f);
  EXPECT_EQ((one == one).zero_mask(), 0);
  EXPECT_EQ((one != two).zero_mask(), 0);
  EXPECT_EQ((one < two).zero_mask(), 0);
  EXPECT_EQ((one <= one).zero_mask(), 0);
  EXPECT_EQ((two > one).zero_mask(), 0);
  EXPECT_EQ((two >= two).zero_mask(), 0);
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
