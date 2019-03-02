#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;
void TestSimpleCase(TensorOptions& options) {
  auto a = randn({2, 3, 4, 5}, options);
  ASSERT_TRUE(a.prod(-4).equal(a.prod(0)));
  ASSERT_TRUE(a.prod(3).equal(a.prod(-1)));
}

void TestExpressionSpecification(TensorOptions& options) {
  auto a = randn({2, 3, 4, 5}, options);
  ASSERT_TRUE(a.unsqueeze(-5).equal(a.unsqueeze(0)));
  ASSERT_TRUE(a.unsqueeze(4).equal(a.unsqueeze(-1)));

  // can unsqueeze scalar
  auto b = randn(1, options);
  b.unsafeGetTensorImpl()->maybe_zero_dim(true);
  ASSERT_TRUE(b.unsqueeze(0).equal(b.unsqueeze(-1)));
}

void TestEmptyTensor(TensorOptions& options) {
  auto a = randn(0, options);
  ASSERT_TRUE(a.prod(0).equal(at::ones({}, options)));
}

void TestScalarVs1Dim1Size(TensorOptions& options) {
  auto a = randn(1, options);
  ASSERT_TRUE(a.prod(0).equal(a.prod(-1)));
  a.unsafeGetTensorImpl()->maybe_zero_dim(true);
  ASSERT_EQ(a.dim(), 0);
  ASSERT_TRUE(a.prod(0).equal(a.prod(-1)));
}

TEST(TestWrapdim, TestWrapdim) {
  manual_seed(123);
  auto options = device(kCPU).dtype(kFloat);

  TestSimpleCase(options);
  TestEmptyTensor(options);
  TestScalarVs1Dim1Size(options);
  TestExpressionSpecification(options);
}
