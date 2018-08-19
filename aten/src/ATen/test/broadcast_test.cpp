#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"
#include "test_seed.h"

using namespace at;

TEST_CASE( "broadcast", "[]" ) {

  manual_seed(123, at::kCPU);

  Type & T = CPU(kFloat);

  // 0) pre-req tests:
  SECTION( "can't expand empty tensor" ) {
    auto empty = randn({0}, T);
    REQUIRE_THROWS(empty.expand({3}));
  }

  // 1) out-place function with 2 args
  SECTION( "out-place function with 2 args" ) {

    SECTION( "basic" ) {
      auto a = randn({3, 1}, T);
      auto b = randn({5}, T);
      std::vector<int64_t> expanded_sizes = {3, 5};
      REQUIRE((a + b).equal(a.expand(expanded_sizes) + b.expand(expanded_sizes)));
    }

    SECTION( "with scalar" ) {
      auto aScalar = ones({1}, T);
      aScalar.get()->maybe_zero_dim(true);
      auto b = randn({3, 5}, T);
      REQUIRE((aScalar + b).equal(aScalar.expand(b.sizes()) + b.expand(b.sizes())));
    }

    SECTION( "old fallback behavior yields error" ) {
      auto a = randn({3, 5}, T);
      auto b = randn({5, 3}, T);
      REQUIRE_THROWS(a + b);
    }

    SECTION( "with mismatched sizes" ) {
      auto a = randn({3, 5}, T);
      auto b = randn({7, 5}, T);
      REQUIRE_THROWS(a + b);
    }
  }

  SECTION( "out-place function with 3 args" ) {

    SECTION( "basic" ) {
      auto a = randn({3, 1, 1}, T);
      auto b = randn({1, 2, 1}, T);
      auto c = randn({1, 1, 5}, T);
      std::vector<int64_t> expanded_sizes = {3, 2, 5};
      REQUIRE((a + b + c).equal(a.expand(expanded_sizes) + b.expand(expanded_sizes) + c.expand(expanded_sizes)));
    }

    SECTION( "with scalar" ) {
      auto aTensorScalar = ones({1}, T);
      aTensorScalar.get()->maybe_zero_dim(true);
      auto b = randn({3, 2, 1}, T);
      auto c = randn({1, 2, 5}, T);
      std::vector<int64_t> expanded_sizes = {3, 2, 5};
      REQUIRE(aTensorScalar.addcmul(b, c).equal(
                aTensorScalar.expand(expanded_sizes).addcmul(b.expand(expanded_sizes), c.expand(expanded_sizes))));
    }

    SECTION( "old fallback behavior yields error" ) {
      auto a = randn({3, 2, 5}, T);
      auto b = randn({2, 3, 5}, T);
      auto c = randn({5, 3, 2}, T);
      REQUIRE_THROWS(a.addcmul(b, c));
    }

    SECTION( "with mismatched sizes" ){
      auto a = randn({3, 2, 5}, T);
      auto b = randn({2, 3, 5}, T);
      auto c = randn({5, 5, 5}, T);
      REQUIRE_THROWS(a.addcmul(b, c));
    }
  }

  SECTION( "in-place function with 2 args" ) {
    SECTION( "basic" ) {
      auto a = randn({3, 5}, T);
      auto b = randn({3, 1}, T);
      REQUIRE((a + b).equal(a + b.expand({3, 5})));
    }

    SECTION( "with scalar" ) {
      auto a = randn({3, 5}, T);
      auto bScalar = ones({1}, T);
      bScalar.get()->maybe_zero_dim(true);
      REQUIRE((a + bScalar).equal(a + bScalar.expand(a.sizes())));
    }

    SECTION( "error: would have to expand inplace arg" ) {
      auto a = randn({1, 5}, T);
      auto b = randn({3, 1}, T);
      REQUIRE_THROWS(a.add_(b));
    }
  }

  SECTION( "in-place function with 3 args" ) {

    auto a = randn({3, 5, 2}, T);
    auto b = randn({3, 1, 2}, T);
    auto c = randn({1, 5, 1}, T);

    SECTION( "basic" ) {
      auto aClone = a.clone();
      REQUIRE(a.addcmul_(b, c).equal(aClone.addcmul_(b.expand(a.sizes()), c.expand(a.sizes()))));
    }

    SECTION( "with scalar" ) {
      auto aClone = a.clone();
      auto bScalar = ones({1}, T);
      bScalar.get()->maybe_zero_dim(true);
      REQUIRE(a.addcmul_(bScalar, c).equal(aClone.addcmul_(bScalar.expand(a.sizes()), c.expand(a.sizes()))));
    }

    SECTION( "error: would have to expand inplace arg" ) {
      auto a = randn({1, 3, 5}, T);
      auto b = randn({4, 1, 1}, T);
      auto c = randn({1, 3, 1}, T);
      REQUIRE_THROWS(a.addcmul_(b, c));
    }
  }

  SECTION( "explicit dim specification" ) {

    auto a = randn({1}, T);
    auto b = randn({5, 3}, T);
    auto c = randn({3, 7}, T);

    SECTION( "basic" ) {
      REQUIRE(a.addmm(b, c).equal(a.expand({5,7}).addmm(b, c)));
    }

    SECTION( "with scalar" ) {
      Tensor aScalar = ones({1}, T);
      aScalar.get()->maybe_zero_dim(true);
      REQUIRE(aScalar.addmm(b, c).equal(aScalar.expand({5, 7}).addmm(b, c)));
    }

    SECTION( "with mismatched sizes" ) {
      auto a = randn({3, 3}, T);
      REQUIRE_THROWS(a.addmm(b, c));
    }
  }
}
