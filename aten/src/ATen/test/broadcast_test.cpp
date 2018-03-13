#include "ATen/ATen.h"
#include "test_assert.h"
#include "test_seed.h"

using namespace at;

int main() {
  manual_seed(123);

  Type & T = CPU(kFloat);

  // 0) pre-req tests:
  // can't expand empty tensor
  {
    auto empty = randn(T, {0});
    ASSERT_THROWS(empty.expand({3}));
  }

  // 1) out-place function with 2 args
  {
    // basic
    auto a = randn(T, {3, 1});
    auto b = randn(T, {5});
    std::vector<int64_t> expanded_sizes = {3, 5};
    ASSERT((a + b).equal(a.expand(expanded_sizes) + b.expand(expanded_sizes)));

    // with scalar
    auto aScalar = ones(T, {1});
    aScalar.get()->maybeScalar(true);
    b = randn(T, {3, 5});
    ASSERT((aScalar + b).equal(aScalar.expand(b.sizes()) + b.expand(b.sizes())));

    // old fallback behavior yields error
    {
      auto a = randn(T, {3, 5});
      auto b = randn(T, {5, 3});
      ASSERT_THROWS(a + b);
    }

    // with mismatched sizes
    {
      auto a = randn(T, {3, 5});
      auto b = randn(T, {7, 5});
      ASSERT_THROWS(a + b);
    }
  }

  // 2) out-place function with 3 args
  {
    // basic
    auto a = randn(T, {3, 1, 1});
    auto b = randn(T, {1, 2, 1});
    auto c = randn(T, {1, 1, 5});
    std::vector<int64_t> expanded_sizes = {3, 2, 5};
    ASSERT((a + b + c).equal(a.expand(expanded_sizes) + b.expand(expanded_sizes) + c.expand(expanded_sizes)));

    // with scalar
    auto aTensorScalar = ones(T, {1});
    aTensorScalar.get()->maybeScalar(true);
    b = randn(T, {3, 2, 1});
    c = randn(T, {1, 2, 5});
    ASSERT(aTensorScalar.addcmul(b, c).equal(
           aTensorScalar.expand(expanded_sizes).addcmul(b.expand(expanded_sizes), c.expand(expanded_sizes))));

    // old fallback behavior yields error
    {
      auto a = randn(T, {3, 2, 5});
      auto b = randn(T, {2, 3, 5});
      auto c = randn(T, {5, 3, 2});
      ASSERT_THROWS(a.addcmul(b, c));
    }

    // with mismatched sizes
    {
      auto c = randn(T, {5, 5, 5});
      ASSERT_THROWS(a.addcmul(b, c));
    }
  }

  // 3) in-place function with 2 args
  {
    // basic
    auto a = randn(T, {3, 5});
    auto b = randn(T, {3, 1});
    ASSERT((a + b).equal(a + b.expand({3, 5})));

    // with scalar
    auto bScalar = ones(T, {1});
    bScalar.get()->maybeScalar(true);
    ASSERT((a + bScalar).equal(a + bScalar.expand(a.sizes())));

    // error: would have to expand inplace arg
    {
      auto a = randn(T, {1, 5});
      auto b = randn(T, {3, 1});
      ASSERT_THROWS(a.add_(b));
    }
  }

  // 4) in-place function with 3 args
  {
    // basic
    auto a = randn(T, {3, 5, 2});
    auto aClone = a.clone();
    auto b = randn(T, {3, 1, 2});
    auto c = randn(T, {1, 5, 1});

    ASSERT(a.addcmul_(b, c).equal(aClone.addcmul_(b.expand(a.sizes()), c.expand(a.sizes()))));

    // with scalar
    auto bScalar = ones(T, {1});
    bScalar.get()->maybeScalar(true);
    ASSERT(a.addcmul_(bScalar, c).equal(aClone.addcmul_(bScalar.expand(a.sizes()), c.expand(a.sizes()))));

    // error: would have to expand inplace arg
    {
      auto a = randn(T, {1, 3, 5});
      auto b = randn(T, {4, 1, 1});
      auto c = randn(T, {1, 3, 1});
      ASSERT_THROWS(a.addcmul_(b, c));
    }
  }

  // explicit dim specification
  {
    // basic
    auto a = randn(T, {1});
    auto b = randn(T, {5, 3});
    auto c = randn(T, {3, 7});
    ASSERT(a.addmm(b, c).equal(a.expand({5,7}).addmm(b, c)));

    // with scalar
    Tensor aScalar = ones(T, {1});
    aScalar.get()->maybeScalar(true);
    ASSERT(aScalar.addmm(b, c).equal(aScalar.expand({5, 7}).addmm(b, c)));

    // with mismatched sizes
    {
      auto a = randn(T, {3, 3});
      ASSERT_THROWS(a.addmm(b, c));
    }
  }

  return 0;
}
