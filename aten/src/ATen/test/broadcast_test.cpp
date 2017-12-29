#include "ATen/ATen.h"
#include "test_assert.h"

using namespace at;

int main() {
  Type & T = CPU(kFloat);

  // 0) pre-req tests:
  // can't expand empty tensor
  try {
    auto empty = T.randn({0});
    empty.expand({3});
    ASSERT(false);
  } catch(std::runtime_error &e) {}

  // 1) out-place function with 2 args
  {
    // basic
    auto a = T.randn({3, 1});
    auto b = T.randn({5});
    std::vector<int64_t> expanded_sizes = {3, 5};
    ASSERT((a + b).equal(a.expand(expanded_sizes) + b.expand(expanded_sizes)));

    // with scalar
    auto aScalar = T.ones({1});
    aScalar.get()->maybeScalar(true);
    b = T.randn({3, 5});
    ASSERT((aScalar + b).equal(aScalar.expand(b.sizes()) + b.expand(b.sizes())));

    // old fallback behavior yields error
    try {
      auto a = T.randn({3, 5});
      auto b = T.randn({5, 3});
      a + b;
      ASSERT(false);
    } catch (std::runtime_error &e) {}

    // with mismatched sizes
    try {
      auto a = T.randn({3, 5});
      auto b = T.randn({7, 5});
      a + b;
      ASSERT(false);
    } catch (std::runtime_error &e) {}
  }

  // 2) out-place function with 3 args
  {
    // basic
    auto a = T.randn({3, 1, 1});
    auto b = T.randn({1, 2, 1});
    auto c = T.randn({1, 1, 5});
    std::vector<int64_t> expanded_sizes = {3, 2, 5};
    ASSERT((a + b + c).equal(a.expand(expanded_sizes) + b.expand(expanded_sizes) + c.expand(expanded_sizes)));

    // with scalar
    auto aTensorScalar = T.ones({1});
    aTensorScalar.get()->maybeScalar(true);
    b = T.randn({3, 2, 1});
    c = T.randn({1, 2, 5});
    ASSERT(aTensorScalar.addcmul(b, c).equal(
           aTensorScalar.expand(expanded_sizes).addcmul(b.expand(expanded_sizes), c.expand(expanded_sizes))));

    // old fallback behavior yields error
    try {
      auto a = T.randn({3, 2, 5});
      auto b = T.randn({2, 3, 5});
      auto c = T.randn({5, 3, 2});
      a.addcmul(b, c);
      ASSERT(false);
    } catch(std::runtime_error &e) {}

    // with mismatched sizes
    try {
      auto c = T.randn({5, 5, 5});
      a.addcmul(b, c);
      ASSERT(false);
    } catch(std::runtime_error &e) {}
  }

  // 3) in-place function with 2 args
  {
    // basic
    auto a = T.randn({3, 5});
    auto b = T.randn({3, 1});
    ASSERT((a + b).equal(a + b.expand({3, 5})));

    // with scalar
    auto bScalar = T.ones({1});
    bScalar.get()->maybeScalar(true);
    ASSERT((a + bScalar).equal(a + bScalar.expand(a.sizes())));

    // error: would have to expand inplace arg
    try {
      auto a = T.randn({1, 5});
      auto b = T.randn({3, 1});
      a.add_(b);
      ASSERT(false);
    } catch(std::runtime_error &e) {}
  }

  // 4) in-place function with 3 args
  {
    // basic
    auto a = T.randn({3, 5, 2});
    auto aClone = a.clone();
    auto b = T.randn({3, 1, 2});
    auto c = T.randn({1, 5, 1});

    ASSERT(a.addcmul_(b, c).equal(aClone.addcmul_(b.expand(a.sizes()), c.expand(a.sizes()))));

    // with scalar
    auto bScalar = T.ones({1});
    bScalar.get()->maybeScalar(true);
    ASSERT(a.addcmul_(bScalar, c).equal(aClone.addcmul_(bScalar.expand(a.sizes()), c.expand(a.sizes()))));

    // error: would have to expand inplace arg
    try {
      auto a = T.randn({1, 3, 5});
      auto b = T.randn({4, 1, 1});
      auto c = T.randn({1, 3, 1});
      a.addcmul_(b, c);
      ASSERT(false);
    } catch(std::runtime_error &e) {}
  }

  // explicit dim specification
  {
    // basic
    auto a = T.randn({1});
    auto b = T.randn({5, 3});
    auto c = T.randn({3, 7});
    ASSERT(a.addmm(b, c).equal(a.expand({5,7}).addmm(b, c)));

    // with scalar
    Tensor aScalar = T.ones({1});
    aScalar.get()->maybeScalar(true);
    ASSERT(aScalar.addmm(b, c).equal(aScalar.expand({5, 7}).addmm(b, c)));

    // with mismatched sizes
    try {
      auto a = T.randn({3, 3});
      a.addmm(b, c);
      ASSERT(false);
    } catch(std::runtime_error &e) {}
  }

  return 0;
}
