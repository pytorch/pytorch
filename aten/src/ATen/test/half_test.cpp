#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <cmath>
#include <type_traits>
#include <ATen/test/test_assert.h>

using namespace at;

TEST(TestHalf, Arithmetic) {
  Half zero = 0;
  Half one = 1;
  ASSERT_EQ(zero + one, one);
  ASSERT_EQ(zero + zero, zero);
  ASSERT_EQ(zero * one, zero);
  ASSERT_EQ(one * one, one);
  ASSERT_EQ(one / one, one);
  ASSERT_EQ(one - one, zero);
  ASSERT_EQ(one - zero, one);
  ASSERT_EQ(zero - one, -one);
  ASSERT_EQ(one + one, Half(2));
  ASSERT_EQ(one + one, 2);
}

TEST(TestHalf, Comparisions) {
  Half zero = 0;
  Half one = 1;
  ASSERT_LT(zero, one);
  ASSERT_LT(zero, 1);
  ASSERT_GT(1, zero);
  ASSERT_GE(0, zero);
  ASSERT_NE(0, one);
  ASSERT_EQ(zero, 0);
  ASSERT_EQ(zero, zero);
  ASSERT_EQ(zero, -zero);
}

TEST(TestHalf, Cast) {
  Half value = 1.5f;
  ASSERT_EQ((int)value, 1);
  ASSERT_EQ((short)value, 1);
  ASSERT_EQ((long long)value, 1LL);
  ASSERT_EQ((float)value, 1.5f);
  ASSERT_EQ((double)value, 1.5);
  ASSERT_EQ((bool)value, true);
  ASSERT_EQ((bool)Half(0.0f), false);
}

TEST(TestHalf, Construction) {
  ASSERT_EQ(Half((short)3), Half(3.0f));
  ASSERT_EQ(Half((unsigned short)3), Half(3.0f));
  ASSERT_EQ(Half(3), Half(3.0f));
  ASSERT_EQ(Half(3U), Half(3.0f));
  ASSERT_EQ(Half(3LL), Half(3.0f));
  ASSERT_EQ(Half(3ULL), Half(3.0f));
  ASSERT_EQ(Half(3.5), Half(3.5f));
}

static std::string to_string(const Half& h) {
  std::stringstream ss;
  ss << h;
  return ss.str();
}

TEST(TestHalf, Half2String) {
  ASSERT_EQ(to_string(Half(3.5f)), "3.5");
  ASSERT_EQ(to_string(Half(-100.0f)), "-100");
}

TEST(TestHalf, HalfNumericLimits) {
  using limits = std::numeric_limits<Half>;
  ASSERT_EQ(limits::lowest(), -65504.0f);
  ASSERT_EQ(limits::max(), 65504.0f);
  ASSERT_GT(limits::min(), 0);
  ASSERT_LT(limits::min(), 1);
  ASSERT_GT(limits::denorm_min(), 0);
  ASSERT_EQ(limits::denorm_min() / 2, 0);
  ASSERT_EQ(limits::infinity(), std::numeric_limits<float>::infinity());
  ASSERT_NE(limits::quiet_NaN(), limits::quiet_NaN());
  ASSERT_NE(limits::signaling_NaN(), limits::signaling_NaN());
}

// Check the declared type of members of numeric_limits<Half> matches
// the declared type of that member on numeric_limits<float>

#define ASSERT_SAME_TYPE(name)                                \
  static_assert(                                              \
      std::is_same<                                           \
          decltype(std::numeric_limits<Half>::name),          \
          decltype(std::numeric_limits<float>::name)>::value, \
      "decltype(" #name ") differs")

ASSERT_SAME_TYPE(is_specialized);
ASSERT_SAME_TYPE(is_signed);
ASSERT_SAME_TYPE(is_integer);
ASSERT_SAME_TYPE(is_exact);
ASSERT_SAME_TYPE(has_infinity);
ASSERT_SAME_TYPE(has_quiet_NaN);
ASSERT_SAME_TYPE(has_signaling_NaN);
ASSERT_SAME_TYPE(has_denorm);
ASSERT_SAME_TYPE(has_denorm_loss);
ASSERT_SAME_TYPE(round_style);
ASSERT_SAME_TYPE(is_iec559);
ASSERT_SAME_TYPE(is_bounded);
ASSERT_SAME_TYPE(is_modulo);
ASSERT_SAME_TYPE(digits);
ASSERT_SAME_TYPE(digits10);
ASSERT_SAME_TYPE(max_digits10);
ASSERT_SAME_TYPE(radix);
ASSERT_SAME_TYPE(min_exponent);
ASSERT_SAME_TYPE(min_exponent10);
ASSERT_SAME_TYPE(max_exponent);
ASSERT_SAME_TYPE(max_exponent10);
ASSERT_SAME_TYPE(traps);
ASSERT_SAME_TYPE(tinyness_before);

TEST(TestHalf, CommonMath) {
  float threshold = 0.00001;
  assert(std::abs(std::lgamma(Half(10.0)) - std::lgamma(10.0f)) <= threshold);
  assert(std::abs(std::exp(Half(1.0)) - std::exp(1.0f)) <= threshold);
  assert(std::abs(std::log(Half(1.0)) - std::log(1.0f)) <= threshold);
  assert(std::abs(std::log10(Half(1000.0)) - std::log10(1000.0f)) <= threshold);
  assert(std::abs(std::log1p(Half(0.0)) - std::log1p(0.0f)) <= threshold);
  assert(std::abs(std::log2(Half(1000.0)) - std::log2(1000.0f)) <= threshold);
  assert(std::abs(std::expm1(Half(1.0)) - std::expm1(1.0f)) <= threshold);
  assert(std::abs(std::cos(Half(0.0)) - std::cos(0.0f)) <= threshold);
  assert(std::abs(std::sin(Half(0.0)) - std::sin(0.0f)) <= threshold);
  assert(std::abs(std::sqrt(Half(100.0)) - std::sqrt(100.0f)) <= threshold);
  assert(std::abs(std::ceil(Half(2.4)) - std::ceil(2.4f)) <= threshold);
  assert(std::abs(std::floor(Half(2.7)) - std::floor(2.7f)) <= threshold);
  assert(std::abs(std::trunc(Half(2.7)) - std::trunc(2.7f)) <= threshold);
  assert(std::abs(std::acos(Half(-1.0)) - std::acos(-1.0f)) <= threshold);
  assert(std::abs(std::cosh(Half(1.0)) - std::cosh(1.0f)) <= threshold);
  assert(std::abs(std::acosh(Half(1.0)) - std::acosh(1.0f)) <= threshold);
  assert(std::abs(std::asin(Half(1.0)) - std::asin(1.0f)) <= threshold);
  assert(std::abs(std::sinh(Half(1.0)) - std::sinh(1.0f)) <= threshold);
  assert(std::abs(std::asinh(Half(1.0)) - std::asinh(1.0f)) <= threshold);
  assert(std::abs(std::tan(Half(0.0)) - std::tan(0.0f)) <= threshold);
  assert(std::abs(std::atan(Half(1.0)) - std::atan(1.0f)) <= threshold);
  assert(std::abs(std::tanh(Half(1.0)) - std::tanh(1.0f)) <= threshold);
  assert(std::abs(std::erf(Half(10.0)) - std::erf(10.0f)) <= threshold);
  assert(std::abs(std::erfc(Half(10.0)) - std::erfc(10.0f)) <= threshold);
  assert(std::abs(std::abs(Half(-3.0)) - std::abs(-3.0f)) <= threshold);
  assert(std::abs(std::round(Half(2.3)) - std::round(2.3f)) <= threshold);
  assert(
      std::abs(std::pow(Half(2.0), Half(10.0)) - std::pow(2.0f, 10.0f)) <=
      threshold);
  assert(
      std::abs(std::atan2(Half(7.0), Half(0.0)) - std::atan2(7.0f, 0.0f)) <=
      threshold);
#ifdef __APPLE__
  // @TODO: can macos do implicit conversion of Half?
  assert(
      std::abs(std::isnan(static_cast<float>(Half(0.0))) - std::isnan(0.0f)) <=
      threshold);
  assert(
      std::abs(std::isinf(static_cast<float>(Half(0.0))) - std::isinf(0.0f)) <=
      threshold);
#else
  assert(std::abs(std::isnan(Half(0.0)) - std::isnan(0.0f)) <= threshold);
  assert(std::abs(std::isinf(Half(0.0)) - std::isinf(0.0f)) <= threshold);
#endif
}
