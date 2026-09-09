#include <gtest/gtest.h>

#include <torch/headeronly/cpu/vec/vec128/vec128_float_neon.h>

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

#if defined(__aarch64__) && !defined(CPU_CAPABILITY_SVE256)
namespace {

using Vec = torch::headeronly::vec::Vectorized<float>;

float Add(float a, float b) {
  return a + b;
}

float SquareRoot(float value) {
  return std::sqrt(value);
}

void ExpectFirstLaneNear(
    const Vec& actual,
    float expected,
    float tolerance = 1e-5f) {
  std::array<float, Vec::size()> output{};
  actual.store(output.data());
  EXPECT_NEAR(output[0], expected, tolerance);
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
  ExpectFirstLaneNear(Vec(1.0f).map2(Vec(2.0f), Add), 3.0f);
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

TEST(TestVec128FloatNeon, TestPublicMethodSurface) {
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

  const Vec half(0.5f);
  const Vec one(1.0f);
  const Vec two(2.0f);
  static_assert(
      torch::headeronly::vec::is_vec_specialized_for<float>::value);
  static_assert(torch::headeronly::vec::is_vec_specialized_for_v<float>);
  static_assert(std::is_same_v<Vec::value_type, float>);
  static_assert(std::is_same_v<Vec::size_type, int>);
  static_assert(Vec::size() == 4);
  ExpectFirstLaneNear(Vec::arange(1.0f, 1.0f), 1.0f);
  ExpectFirstLaneNear(Vec::set(Vec(0.0f), one, 1), 1.0f);
  ExpectFirstLaneNear(Vec::loadu(values), 0.5f);
  EXPECT_EQ(Vec(0.0f).zero_mask(), 0b1111);
  EXPECT_TRUE(Vec(std::numeric_limits<float>::quiet_NaN()).has_inf_nan());
  EXPECT_EQ(
      Vec(std::numeric_limits<float>::quiet_NaN()).isnan().zero_mask(), 0);
  ExpectFirstLaneNear(half.map(SquareRoot), std::sqrt(0.5f));
  ExpectFirstLaneNear(Vec(-0.5f).abs(), 0.5f);
  ExpectFirstLaneNear(Vec(-0.5f).angle(), c10::pi<float>);
  ExpectFirstLaneNear(half.real(), 0.5f);
  ExpectFirstLaneNear(half.imag(), 0.0f);
  ExpectFirstLaneNear(half.conj(), 0.5f);

  ExpectFirstLaneNear(half.acos(), std::acos(0.5f));
  ExpectFirstLaneNear(two.acosh(), std::acosh(2.0f));
  ExpectFirstLaneNear(half.asin(), std::asin(0.5f));
  ExpectFirstLaneNear(half.asinh(), std::asinh(0.5f));
  ExpectFirstLaneNear(half.atan(), std::atan(0.5f));
  ExpectFirstLaneNear(half.atanh(), std::atanh(0.5f));
  ExpectFirstLaneNear(half.atan2(two), std::atan2(0.5f, 2.0f));
  ExpectFirstLaneNear(two.copysign(Vec(-1.0f)), -2.0f);
  ExpectFirstLaneNear(half.erfc(), std::erfc(0.5f));
  ExpectFirstLaneNear(Vec(0.0f).erfinv(), 0.0f);
  ExpectFirstLaneNear(half.exp(), std::exp(0.5f));
  ExpectFirstLaneNear(half.exp2(), std::exp2(0.5f));
  ExpectFirstLaneNear(half.expm1(), std::expm1(0.5f));
  ExpectFirstLaneNear(Vec(1.5f).frac(), 0.5f);
  ExpectFirstLaneNear(Vec(5.5f).fmod(two), 1.5f);
  ExpectFirstLaneNear(Vec(3.0f).hypot(Vec(4.0f)), 5.0f);
  ExpectFirstLaneNear(Vec(0.0f).i0(), 1.0f);
  ExpectFirstLaneNear(Vec(0.0f).i0e(), 1.0f);
  ExpectFirstLaneNear(Vec(1.0f).digamma(), -0.57721566f, 1e-4f);
  ExpectFirstLaneNear(Vec(1.0f).igamma(one), 1.0f - std::exp(-1.0f), 1e-4f);
  ExpectFirstLaneNear(Vec(1.0f).igammac(one), std::exp(-1.0f), 1e-4f);
  ExpectFirstLaneNear(two.log(), std::log(2.0f));
  ExpectFirstLaneNear(two.log10(), std::log10(2.0f));
  ExpectFirstLaneNear(two.log1p(), std::log1p(2.0f));
  ExpectFirstLaneNear(two.log2(), 1.0f);
  ExpectFirstLaneNear(Vec(1.0f).nextafter(two), std::nextafter(1.0f, 2.0f));
  ExpectFirstLaneNear(half.sin(), std::sin(0.5f));
  ExpectFirstLaneNear(half.sinh(), std::sinh(0.5f));
  ExpectFirstLaneNear(half.cos(), std::cos(0.5f));
  ExpectFirstLaneNear(half.cosh(), std::cosh(0.5f));
  ExpectFirstLaneNear(Vec(1.5f).ceil(), 2.0f);
  ExpectFirstLaneNear(Vec(1.5f).floor(), 1.0f);
  ExpectFirstLaneNear(Vec(1.5f).round(), 2.0f);
  ExpectFirstLaneNear(Vec(1.5f).trunc(), 1.0f);
  ExpectFirstLaneNear(Vec(1.0f).tan(), std::tan(1.0f));
  ExpectFirstLaneNear(half.tanh(), std::tanh(0.5f));
  ExpectFirstLaneNear(Vec(1.0f).lgamma(), 0.0f);
  ExpectFirstLaneNear(Vec(4.0f).sqrt(), 2.0f);
  ExpectFirstLaneNear(two.reciprocal(), 0.5f);
  ExpectFirstLaneNear(Vec(4.0f).rsqrt(), 0.5f);
  ExpectFirstLaneNear(two.pow(Vec(3.0f)), 8.0f);
  ExpectFirstLaneNear(two.neg(), -2.0f);

  ExpectFirstLaneNear(one.eq(one), 1.0f);
  ExpectFirstLaneNear(one.ne(two), 1.0f);
  ExpectFirstLaneNear(two.gt(one), 1.0f);
  ExpectFirstLaneNear(two.ge(two), 1.0f);
  ExpectFirstLaneNear(one.lt(two), 1.0f);
  ExpectFirstLaneNear(one.le(one), 1.0f);
  EXPECT_EQ((one == one).zero_mask(), 0);
  EXPECT_EQ((one != two).zero_mask(), 0);
  EXPECT_EQ((one < two).zero_mask(), 0);
  EXPECT_EQ((one <= one).zero_mask(), 0);
  EXPECT_EQ((two > one).zero_mask(), 0);
  EXPECT_EQ((two >= two).zero_mask(), 0);

  ExpectFirstLaneNear(torch::headeronly::vec::operator+(one, two), 3.0f);
  ExpectFirstLaneNear(torch::headeronly::vec::operator-(two, one), 1.0f);
  ExpectFirstLaneNear(torch::headeronly::vec::operator*(two, two), 4.0f);
  ExpectFirstLaneNear(torch::headeronly::vec::operator/(two, two), 1.0f);
  ExpectFirstLaneNear(torch::headeronly::vec::maximum(one, two), 2.0f);
  ExpectFirstLaneNear(torch::headeronly::vec::minimum(one, two), 1.0f);
  ExpectFirstLaneNear(torch::headeronly::vec::clamp(two, Vec(0.0f), one), 1.0f);
  ExpectFirstLaneNear(torch::headeronly::vec::clamp_max(two, one), 1.0f);
  ExpectFirstLaneNear(torch::headeronly::vec::clamp_min(one, two), 2.0f);
  ExpectFirstLaneNear(torch::headeronly::vec::fmadd(two, two, one), 5.0f);
  ExpectFirstLaneNear(torch::headeronly::vec::fnmadd(two, two, one), -3.0f);
  ExpectFirstLaneNear(torch::headeronly::vec::fmsub(two, two, one), 3.0f);
  ExpectFirstLaneNear(torch::headeronly::vec::fnmsub(two, two, one), -5.0f);
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
