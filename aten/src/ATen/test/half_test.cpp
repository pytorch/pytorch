#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <ATen/ATen.h>
#include <iostream>
#include <sstream>

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

TEST_CASE( "half cmath ", "[]" ) {
  // Relies on implicit conversion to float
  using limits = std::numeric_limits<Half>;
  REQUIRE(std::isnan(limits::quiet_NaN()));
  REQUIRE(std::isnan(limits::signaling_NaN()));
  REQUIRE(!std::isinf(limits::quiet_NaN()));
  REQUIRE(!std::isinf(limits::signaling_NaN()));
  REQUIRE(std::isinf(limits::infinity()));
  REQUIRE(!std::isnan(limits::infinity()));
}
