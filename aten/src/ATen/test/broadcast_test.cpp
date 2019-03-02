
#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;

// can't expand empty tensor
void TestEmptyTensor(TensorOptions& options) {
  auto empty = randn({0}, options);
  ASSERT_ANY_THROW(empty.expand({3}));
}

// out-place function with 2 args
void TestOut2Basic(TensorOptions& options) {
  auto a = randn({3, 1}, options);
  auto b = randn({5}, options);
  std::vector<int64_t> expanded_sizes = {3, 5};
  ASSERT_TRUE(
      (a + b).equal(a.expand(expanded_sizes) + b.expand(expanded_sizes)));
}

// with scalar
void TestOut2WithScalar(TensorOptions& options) {
  auto aScalar = ones({1}, options);
  aScalar.unsafeGetTensorImpl()->maybe_zero_dim(true);
  auto b = randn({3, 5}, options);
  ASSERT_TRUE(
      (aScalar + b).equal(aScalar.expand(b.sizes()) + b.expand(b.sizes())));
}

// old fallback behavior yields error
void TestOut2OldFallback(TensorOptions& options) {
  auto a = randn({3, 5}, options);
  auto b = randn({5, 3}, options);
  ASSERT_ANY_THROW(a + b);
}

// with mismatched sizes
void TestOut2MismatchedSizes(TensorOptions& options) {
  auto a = randn({3, 5}, options);
  auto b = randn({7, 5}, options);
  ASSERT_ANY_THROW(a + b);
}

// out-place function with 3 args
void TestOut3Basic(TensorOptions& options) {
  auto a = randn({3, 1, 1}, options);
  auto b = randn({1, 2, 1}, options);
  auto c = randn({1, 1, 5}, options);
  std::vector<int64_t> expanded_sizes = {3, 2, 5};
  ASSERT_TRUE((a + b + c).equal(
      a.expand(expanded_sizes) + b.expand(expanded_sizes) +
      c.expand(expanded_sizes)));
}

// with scalar
void TestOut3WithScalar(TensorOptions& options) {
  auto aTensorScalar = ones({1}, options);
  aTensorScalar.unsafeGetTensorImpl()->maybe_zero_dim(true);
  auto b = randn({3, 2, 1}, options);
  auto c = randn({1, 2, 5}, options);
  std::vector<int64_t> expanded_sizes = {3, 2, 5};
  ASSERT_TRUE(aTensorScalar.addcmul(b, c).equal(
      aTensorScalar.expand(expanded_sizes)
          .addcmul(b.expand(expanded_sizes), c.expand(expanded_sizes))));
}

// old fallback behavior yields error
void TestOut3OldFallback(TensorOptions& options) {
  auto a = randn({3, 2, 5}, options);
  auto b = randn({2, 3, 5}, options);
  auto c = randn({5, 3, 2}, options);
  ASSERT_ANY_THROW(a.addcmul(b, c));
}

// with mismatched sizes
void TestOut3MismatchedSizes(TensorOptions& options) {
  auto a = randn({3, 2, 5}, options);
  auto b = randn({2, 3, 5}, options);
  auto c = randn({5, 5, 5}, options);
  ASSERT_ANY_THROW(a.addcmul(b, c));
}

// in-place function with 2 args
void TestIn2Basic(TensorOptions& options) {
  auto a = randn({3, 5}, options);
  auto b = randn({3, 1}, options);
  ASSERT_TRUE((a + b).equal(a + b.expand({3, 5})));
}

// with scalar
void TestIn2WithScalar(TensorOptions& options) {
  auto a = randn({3, 5}, options);
  auto bScalar = ones({1}, options);
  bScalar.unsafeGetTensorImpl()->maybe_zero_dim(true);
  ASSERT_TRUE((a + bScalar).equal(a + bScalar.expand(a.sizes())));
}

// error: would have to expand inplace arg
void TestIn2ExpandError(TensorOptions& options) {
  auto a = randn({1, 5}, options);
  auto b = randn({3, 1}, options);
  ASSERT_ANY_THROW(a.add_(b));
}

// in-place function with 3 args
void TestIn3Basic(TensorOptions& options) {
  auto a = randn({3, 5, 2}, options);
  auto b = randn({3, 1, 2}, options);
  auto c = randn({1, 5, 1}, options);
  auto aClone = a.clone();
  ASSERT_TRUE(a.addcmul_(b, c).equal(
      aClone.addcmul_(b.expand(a.sizes()), c.expand(a.sizes()))));
}

// with scalar
void TestIn3WithScalar(TensorOptions& options) {
  auto a = randn({3, 5, 2}, options);
  auto b = randn({3, 1, 2}, options);
  auto c = randn({1, 5, 1}, options);
  auto aClone = a.clone();
  auto bScalar = ones({1}, options);
  bScalar.unsafeGetTensorImpl()->maybe_zero_dim(true);
  ASSERT_TRUE(a.addcmul_(bScalar, c)
                  .equal(aClone.addcmul_(
                      bScalar.expand(a.sizes()), c.expand(a.sizes()))));
}

// error: would have to expand inplace arg
void TestIn3ExpandError(TensorOptions& options) {
  auto a = randn({1, 3, 5}, options);
  auto b = randn({4, 1, 1}, options);
  auto c = randn({1, 3, 1}, options);
  ASSERT_ANY_THROW(a.addcmul_(b, c));
}

// explicit dim specification
void TestExplicitDimBasic(TensorOptions& options) {
  auto a = randn({1}, options);
  auto b = randn({5, 3}, options);
  auto c = randn({3, 7}, options);
  ASSERT_TRUE(a.addmm(b, c).equal(a.expand({5, 7}).addmm(b, c)));
}

// with scalar
void TestExplicitDimWithScalar(TensorOptions& options) {
  auto a = randn({1}, options);
  auto b = randn({5, 3}, options);
  auto c = randn({3, 7}, options);
  Tensor aScalar = ones({1}, options);
  aScalar.unsafeGetTensorImpl()->maybe_zero_dim(true);
  ASSERT_TRUE(aScalar.addmm(b, c).equal(aScalar.expand({5, 7}).addmm(b, c)));
}

// with mismatched sizes
void TestExplicitDimWithMismatchedSizes(TensorOptions& options) {
  auto b = randn({5, 3}, options);
  auto c = randn({3, 7}, options);
  auto a = randn({3, 3}, options);
  ASSERT_ANY_THROW(a.addmm(b, c));
}

TEST(BroadcastTest, Broadcast) {
  manual_seed(123);
  TensorOptions options = device(kCPU).dtype(kFloat);

  TestEmptyTensor(options);

  TestOut2Basic(options);
  TestOut2WithScalar(options);
  TestOut2OldFallback(options);
  TestOut2MismatchedSizes(options);

  TestOut3Basic(options);
  TestOut3WithScalar(options);
  TestOut3OldFallback(options);
  TestOut3MismatchedSizes(options);

  TestIn2Basic(options);
  TestIn2WithScalar(options);
  TestIn2ExpandError(options);

  TestIn3Basic(options);
  TestIn3WithScalar(options);
  TestIn3ExpandError(options);

  TestExplicitDimBasic(options);
  TestExplicitDimWithScalar(options);
  TestExplicitDimWithMismatchedSizes(options);
}
