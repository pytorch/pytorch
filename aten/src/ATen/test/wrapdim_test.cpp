#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"
#include "test_seed.h"

using namespace at;

TEST_CASE( "wrapdim test", "[]" ) {
  manual_seed(123, at::kCPU);

  Type & T = CPU(kFloat);

  SECTION( "simple case" ) {
    auto a = randn({2, 3, 4, 5}, T);
    REQUIRE(a.prod(-4).equal(a.prod(0)));
    REQUIRE(a.prod(3).equal(a.prod(-1)));
  }

  SECTION( "expression specification" ) {
    auto a = randn({2, 3, 4, 5}, T);
    REQUIRE(a.unsqueeze(-5).equal(a.unsqueeze(0)));
    REQUIRE(a.unsqueeze(4).equal(a.unsqueeze(-1)));

    // can unsqueeze scalar
    auto b = randn(1, T);
    b.get()->maybe_zero_dim(true);
    REQUIRE(b.unsqueeze(0).equal(b.unsqueeze(-1)));
  }

  SECTION( "empty tensor" ) {
    auto a = randn(0, T);
    REQUIRE(a.prod(0).equal(at::ones({}, T)));
  }

  SECTION( "scalar vs 1-dim, 1-size" ) {
    auto a = randn(1, T);
    REQUIRE(a.prod(0).equal(a.prod(-1)));
    a.get()->maybe_zero_dim(true);
    REQUIRE(a.get()->dim() == 0);
    REQUIRE(a.prod(0).equal(a.prod(-1)));
  }
}
