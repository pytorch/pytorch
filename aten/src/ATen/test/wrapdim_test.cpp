#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;
void TestSimpleCase(DeprecatedTypeProperties& T) {
  auto a = randn({2, 3, 4, 5}, T);
  ASSERT_TRUE(a.prod(-4).equal(a.prod(0)));
  ASSERT_TRUE(a.prod(3).equal(a.prod(-1)));
}

void TestExpressionSpecification(DeprecatedTypeProperties& T) {
  auto a = randn({2, 3, 4, 5}, T);
  ASSERT_TRUE(a.unsqueeze(-5).equal(a.unsqueeze(0)));
  ASSERT_TRUE(a.unsqueeze(4).equal(a.unsqueeze(-1)));

  // can unsqueeze scalar
  auto b = randn({}, T);
  ASSERT_TRUE(b.unsqueeze(0).equal(b.unsqueeze(-1)));
}

void TestEmptyTensor(DeprecatedTypeProperties& T) {
  auto a = randn(0, T);
  ASSERT_TRUE(a.prod(0).equal(at::ones({}, T)));
}

void TestScalarVs1Dim1Size(DeprecatedTypeProperties& T) {
  auto a = randn(1, T);
  ASSERT_TRUE(a.prod(0).equal(a.prod(-1)));
  a.resize_({});
  ASSERT_EQ(a.dim(), 0);
  ASSERT_TRUE(a.prod(0).equal(a.prod(-1)));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TestWrapdim, TestWrapdim) {
  manual_seed(123);
  DeprecatedTypeProperties& T = CPU(kFloat);

  TestSimpleCase(T);
  TestEmptyTensor(T);
  TestScalarVs1Dim1Size(T);
  TestExpressionSpecification(T);
}
