#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <ATen/ATen.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <cmath>
#include <type_traits>
#include "test_seed.h"
#include "test_assert.h"

using namespace at;

TEST_CASE( "half arithmetic", "[]" ) {
  Half zero = 0;
  Half one = 1;
  REQUIRE(zero + one == one);
  REQUIRE(zero + zero == zero);
  REQUIRE(zero * one == zero);
  REQUIRE(one * one == one);
  REQUIRE(one / one == one);
  REQUIRE(one - one == zero);
  REQUIRE(one - zero == one);
  REQUIRE(zero - one == -one);
  REQUIRE(one + one == Half(2));
  REQUIRE(one + one == 2);
}

TEST_CASE( "half comparisons", "[]" ) {
  Half zero = 0;
  Half one = 1;
  REQUIRE(zero < one);
  REQUIRE(zero < 1);
  REQUIRE(1 > zero);
  REQUIRE(0 >= zero);
  REQUIRE(0 != one);
  REQUIRE(zero == 0);
  REQUIRE(zero == zero);
  REQUIRE(zero == -zero);
}

TEST_CASE( "half cast", "[]" ) {
  Half value = 1.5f;
  REQUIRE((int)value == 1);
  REQUIRE((short)value == 1);
  REQUIRE((long long)value == 1LL);
  REQUIRE((float)value == 1.5f);
  REQUIRE((double)value == 1.5);
  REQUIRE((bool)value == true);
  REQUIRE((bool)Half(0.0f) == false);
}

TEST_CASE( "half construction", "[]" ) {
  REQUIRE(Half((short)3) == Half(3.0f));
  REQUIRE(Half((unsigned short)3) == Half(3.0f));
  REQUIRE(Half(3) == Half(3.0f));
  REQUIRE(Half(3U) == Half(3.0f));
  REQUIRE(Half(3LL) == Half(3.0f));
  REQUIRE(Half(3ULL) == Half(3.0f));
  REQUIRE(Half(3.5) == Half(3.5f));
}

static std::string to_string(const Half& h) {
  std::stringstream ss;
  ss << h;
  return ss.str();
}

TEST_CASE( "half to string", "[]" ) {
  REQUIRE(to_string(Half(3.5f)) == "3.5");
  REQUIRE(to_string(Half(-100.0f)) == "-100");
}

TEST_CASE( "half numeric limits", "[]" ) {
  using limits = std::numeric_limits<Half>;
  REQUIRE(limits::lowest() == -65504.0f);
  REQUIRE(limits::max() == 65504.0f);
  REQUIRE(limits::min() > 0);
  REQUIRE(limits::min() < 1);
  REQUIRE(limits::denorm_min() > 0);
  REQUIRE(limits::denorm_min() / 2  == 0);
  REQUIRE(limits::infinity() == std::numeric_limits<float>::infinity());
  REQUIRE(limits::quiet_NaN() != limits::quiet_NaN());
  REQUIRE(limits::signaling_NaN() != limits::signaling_NaN());
}

// Check the declared type of members of numeric_limits<Half> matches
// the declared type of that member on numeric_limits<float>

#define ASSERT_SAME_TYPE(name) \
  static_assert( \
      std::is_same< \
          decltype(std::numeric_limits<Half>::name), \
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

TEST_CASE( "half common math functions test", "[]" ) {
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
  assert(std::abs(std::pow(Half(2.0), Half(10.0)) - std::pow(2.0f, 10.0f)) <= threshold);
  assert(std::abs(std::atan2(Half(7.0), Half(0.0)) - std::atan2(7.0f, 0.0f)) <= threshold);
  #ifdef __APPLE__
    // @TODO: can macos do implicit conversion of Half?
    assert(std::abs(std::isnan(static_cast<float>(Half(0.0))) - std::isnan(0.0f)) <= threshold);
    assert(std::abs(std::isinf(static_cast<float>(Half(0.0))) - std::isinf(0.0f)) <= threshold);
  #else
    assert(std::abs(std::isnan(Half(0.0)) - std::isnan(0.0f)) <= threshold);
    assert(std::abs(std::isinf(Half(0.0)) - std::isinf(0.0f)) <= threshold);
  #endif
}