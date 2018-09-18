#define CATCH_CONFIG_MAIN
#include "catch_utils.hpp"

#include "ATen/ATen.h"
#include "test_seed.h"

using namespace at;

CATCH_TEST_CASE( "wrapdim test", "[]" ) {
  manual_seed(123, at::kCPU);

  Type & T = CPU(kFloat);

  CATCH_SECTION( "simple case" ) {
    auto a = randn({2, 3, 4, 5}, T);
    CATCH_REQUIRE(a.prod(-4).equal(a.prod(0)));
    CATCH_REQUIRE(a.prod(3).equal(a.prod(-1)));
  }

  CATCH_SECTION( "expression specification" ) {
    auto a = randn({2, 3, 4, 5}, T);
    CATCH_REQUIRE(a.unsqueeze(-5).equal(a.unsqueeze(0)));
    CATCH_REQUIRE(a.unsqueeze(4).equal(a.unsqueeze(-1)));

    // can unsqueeze scalar
    auto b = randn(1, T);
    b.unsafeGetTensorImpl()->maybe_zero_dim(true);
    CATCH_REQUIRE(b.unsqueeze(0).equal(b.unsqueeze(-1)));
  }

  CATCH_SECTION( "empty tensor" ) {
    auto a = randn(0, T);
    CATCH_REQUIRE(a.prod(0).equal(at::ones({}, T)));
  }

  CATCH_SECTION( "scalar vs 1-dim, 1-size" ) {
    auto a = randn(1, T);
    CATCH_REQUIRE(a.prod(0).equal(a.prod(-1)));
    a.unsafeGetTensorImpl()->maybe_zero_dim(true);
    CATCH_REQUIRE(a.dim() == 0);
    CATCH_REQUIRE(a.prod(0).equal(a.prod(-1)));
  }
}
