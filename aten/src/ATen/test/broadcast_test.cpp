#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"
#include "test_seed.h"

using namespace at;

TEST_CASE( "broadcast", "[]" ) {

  manual_seed(123, at::Backend::CPU);

  Type & T = CPU(kFloat);

  // 0) pre-req tests:
  SECTION( "can't expand empty tensor" ) {
    auto empty = randn(T, {0});
    REQUIRE_THROWS(empty.expand({3}));
  }

  // 1) out-place function with 2 args
  SECTION( "out-place function with 2 args" ) {

    SECTION( "basic" ) {
      auto a = randn(T, {3, 1});
      auto b = randn(T, {5});
      std::vector<int64_t> expanded_sizes = {3, 5};
      REQUIRE((a + b).equal(a.expand(expanded_sizes) + b.expand(expanded_sizes)));
    }

    SECTION( "with scalar" ) {
      auto aScalar = ones(T, {1});
      aScalar.get()->maybeScalar(true);
      auto b = randn(T, {3, 5});
      REQUIRE((aScalar + b).equal(aScalar.expand(b.sizes()) + b.expand(b.sizes())));
    }

    SECTION( "old fallback behavior yields error" ) {
      auto a = randn(T, {3, 5});
      auto b = randn(T, {5, 3});
      REQUIRE_THROWS(a + b);
    }

    SECTION( "with mismatched sizes" ) {
      auto a = randn(T, {3, 5});
      auto b = randn(T, {7, 5});
      REQUIRE_THROWS(a + b);
    }
  }

  SECTION( "out-place function with 3 args" ) {

    SECTION( "basic" ) {
      auto a = randn(T, {3, 1, 1});
      auto b = randn(T, {1, 2, 1});
      auto c = randn(T, {1, 1, 5});
      std::vector<int64_t> expanded_sizes = {3, 2, 5};
      REQUIRE((a + b + c).equal(a.expand(expanded_sizes) + b.expand(expanded_sizes) + c.expand(expanded_sizes)));
    }

    SECTION( "with scalar" ) {
      auto aTensorScalar = ones(T, {1});
      aTensorScalar.get()->maybeScalar(true);
      auto b = randn(T, {3, 2, 1});
      auto c = randn(T, {1, 2, 5});
      std::vector<int64_t> expanded_sizes = {3, 2, 5};
      REQUIRE(aTensorScalar.addcmul(b, c).equal(
                aTensorScalar.expand(expanded_sizes).addcmul(b.expand(expanded_sizes), c.expand(expanded_sizes))));
    }

    SECTION( "old fallback behavior yields error" ) {
      auto a = randn(T, {3, 2, 5});
      auto b = randn(T, {2, 3, 5});
      auto c = randn(T, {5, 3, 2});
      REQUIRE_THROWS(a.addcmul(b, c));
    }

    SECTION( "with mismatched sizes" ){
      auto a = randn(T, {3, 2, 5});
      auto b = randn(T, {2, 3, 5});
      auto c = randn(T, {5, 5, 5});
      REQUIRE_THROWS(a.addcmul(b, c));
    }
  }

  SECTION( "in-place function with 2 args" ) {
    SECTION( "basic" ) {
      auto a = randn(T, {3, 5});
      auto b = randn(T, {3, 1});
      REQUIRE((a + b).equal(a + b.expand({3, 5})));
    }

    SECTION( "with scalar" ) {
      auto a = randn(T, {3, 5});
      auto bScalar = ones(T, {1});
      bScalar.get()->maybeScalar(true);
      REQUIRE((a + bScalar).equal(a + bScalar.expand(a.sizes())));
    }

    SECTION( "error: would have to expand inplace arg" ) {
      auto a = randn(T, {1, 5});
      auto b = randn(T, {3, 1});
      REQUIRE_THROWS(a.add_(b));
    }
  }

  SECTION( "in-place function with 3 args" ) {

    auto a = randn(T, {3, 5, 2});
    auto b = randn(T, {3, 1, 2});
    auto c = randn(T, {1, 5, 1});

    SECTION( "basic" ) {
      auto aClone = a.clone();
      REQUIRE(a.addcmul_(b, c).equal(aClone.addcmul_(b.expand(a.sizes()), c.expand(a.sizes()))));
    }

    SECTION( "with scalar" ) {
      auto aClone = a.clone();
      auto bScalar = ones(T, {1});
      bScalar.get()->maybeScalar(true);
      REQUIRE(a.addcmul_(bScalar, c).equal(aClone.addcmul_(bScalar.expand(a.sizes()), c.expand(a.sizes()))));
    }

    SECTION( "error: would have to expand inplace arg" ) {
      auto a = randn(T, {1, 3, 5});
      auto b = randn(T, {4, 1, 1});
      auto c = randn(T, {1, 3, 1});
      REQUIRE_THROWS(a.addcmul_(b, c));
    }
  }

  SECTION( "explicit dim specification" ) {

    auto a = randn(T, {1});
    auto b = randn(T, {5, 3});
    auto c = randn(T, {3, 7});

    SECTION( "basic" ) {
      REQUIRE(a.addmm(b, c).equal(a.expand({5,7}).addmm(b, c)));
    }

    SECTION( "with scalar" ) {
      Tensor aScalar = ones(T, {1});
      aScalar.get()->maybeScalar(true);
      REQUIRE(aScalar.addmm(b, c).equal(aScalar.expand({5, 7}).addmm(b, c)));
    }

    SECTION( "with mismatched sizes" ) {
      auto a = randn(T, {3, 3});
      REQUIRE_THROWS(a.addmm(b, c));
    }
  }
}
