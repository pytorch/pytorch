#define CATCH_CONFIG_MAIN
#include "catch_utils.hpp"

#include "ATen/ATen.h"
#include "test_seed.h"

using namespace at;

CATCH_TEST_CASE( "broadcast", "[]" ) {

  manual_seed(123, at::kCPU);

  Type & T = CPU(kFloat);

  // 0) pre-req tests:
  CATCH_SECTION( "can't expand empty tensor" ) {
    auto empty = randn({0}, T);
    _CATCH_REQUIRE_THROWS(empty.expand({3}));
  }

  // 1) out-place function with 2 args
  CATCH_SECTION( "out-place function with 2 args" ) {

    CATCH_SECTION( "basic" ) {
      auto a = randn({3, 1}, T);
      auto b = randn({5}, T);
      std::vector<int64_t> expanded_sizes = {3, 5};
      CATCH_REQUIRE((a + b).equal(a.expand(expanded_sizes) + b.expand(expanded_sizes)));
    }

    CATCH_SECTION( "with scalar" ) {
      auto aScalar = ones({1}, T);
      aScalar.unsafeGetTensorImpl()->maybe_zero_dim(true);
      auto b = randn({3, 5}, T);
      CATCH_REQUIRE((aScalar + b).equal(aScalar.expand(b.sizes()) + b.expand(b.sizes())));
    }

    CATCH_SECTION( "old fallback behavior yields error" ) {
      auto a = randn({3, 5}, T);
      auto b = randn({5, 3}, T);
      _CATCH_REQUIRE_THROWS(a + b);
    }

    CATCH_SECTION( "with mismatched sizes" ) {
      auto a = randn({3, 5}, T);
      auto b = randn({7, 5}, T);
      _CATCH_REQUIRE_THROWS(a + b);
    }
  }

  CATCH_SECTION( "out-place function with 3 args" ) {

    CATCH_SECTION( "basic" ) {
      auto a = randn({3, 1, 1}, T);
      auto b = randn({1, 2, 1}, T);
      auto c = randn({1, 1, 5}, T);
      std::vector<int64_t> expanded_sizes = {3, 2, 5};
      CATCH_REQUIRE((a + b + c).equal(a.expand(expanded_sizes) + b.expand(expanded_sizes) + c.expand(expanded_sizes)));
    }

    CATCH_SECTION( "with scalar" ) {
      auto aTensorScalar = ones({1}, T);
      aTensorScalar.unsafeGetTensorImpl()->maybe_zero_dim(true);
      auto b = randn({3, 2, 1}, T);
      auto c = randn({1, 2, 5}, T);
      std::vector<int64_t> expanded_sizes = {3, 2, 5};
      CATCH_REQUIRE(aTensorScalar.addcmul(b, c).equal(
                aTensorScalar.expand(expanded_sizes).addcmul(b.expand(expanded_sizes), c.expand(expanded_sizes))));
    }

    CATCH_SECTION( "old fallback behavior yields error" ) {
      auto a = randn({3, 2, 5}, T);
      auto b = randn({2, 3, 5}, T);
      auto c = randn({5, 3, 2}, T);
      _CATCH_REQUIRE_THROWS(a.addcmul(b, c));
    }

    CATCH_SECTION( "with mismatched sizes" ){
      auto a = randn({3, 2, 5}, T);
      auto b = randn({2, 3, 5}, T);
      auto c = randn({5, 5, 5}, T);
      _CATCH_REQUIRE_THROWS(a.addcmul(b, c));
    }
  }

  CATCH_SECTION( "in-place function with 2 args" ) {
    CATCH_SECTION( "basic" ) {
      auto a = randn({3, 5}, T);
      auto b = randn({3, 1}, T);
      CATCH_REQUIRE((a + b).equal(a + b.expand({3, 5})));
    }

    CATCH_SECTION( "with scalar" ) {
      auto a = randn({3, 5}, T);
      auto bScalar = ones({1}, T);
      bScalar.unsafeGetTensorImpl()->maybe_zero_dim(true);
      CATCH_REQUIRE((a + bScalar).equal(a + bScalar.expand(a.sizes())));
    }

    CATCH_SECTION( "error: would have to expand inplace arg" ) {
      auto a = randn({1, 5}, T);
      auto b = randn({3, 1}, T);
      _CATCH_REQUIRE_THROWS(a.add_(b));
    }
  }

  CATCH_SECTION( "in-place function with 3 args" ) {

    auto a = randn({3, 5, 2}, T);
    auto b = randn({3, 1, 2}, T);
    auto c = randn({1, 5, 1}, T);

    CATCH_SECTION( "basic" ) {
      auto aClone = a.clone();
      CATCH_REQUIRE(a.addcmul_(b, c).equal(aClone.addcmul_(b.expand(a.sizes()), c.expand(a.sizes()))));
    }

    CATCH_SECTION( "with scalar" ) {
      auto aClone = a.clone();
      auto bScalar = ones({1}, T);
      bScalar.unsafeGetTensorImpl()->maybe_zero_dim(true);
      CATCH_REQUIRE(a.addcmul_(bScalar, c).equal(aClone.addcmul_(bScalar.expand(a.sizes()), c.expand(a.sizes()))));
    }

    CATCH_SECTION( "error: would have to expand inplace arg" ) {
      auto a = randn({1, 3, 5}, T);
      auto b = randn({4, 1, 1}, T);
      auto c = randn({1, 3, 1}, T);
      _CATCH_REQUIRE_THROWS(a.addcmul_(b, c));
    }
  }

  CATCH_SECTION( "explicit dim specification" ) {

    auto a = randn({1}, T);
    auto b = randn({5, 3}, T);
    auto c = randn({3, 7}, T);

    CATCH_SECTION( "basic" ) {
      CATCH_REQUIRE(a.addmm(b, c).equal(a.expand({5,7}).addmm(b, c)));
    }

    CATCH_SECTION( "with scalar" ) {
      Tensor aScalar = ones({1}, T);
      aScalar.unsafeGetTensorImpl()->maybe_zero_dim(true);
      CATCH_REQUIRE(aScalar.addmm(b, c).equal(aScalar.expand({5, 7}).addmm(b, c)));
    }

    CATCH_SECTION( "with mismatched sizes" ) {
      auto a = randn({3, 3}, T);
      _CATCH_REQUIRE_THROWS(a.addmm(b, c));
    }
  }
}
