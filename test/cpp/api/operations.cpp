#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>
struct OperationTest : torch::test::SeedingFixture {
 protected:
  void SetUp() override {}

  const int TEST_AMOUNT = 10;
};

TEST_F(OperationTest, Lerp) {
  for (auto i = 0; i < TEST_AMOUNT; i++) {
    // test lerp_kernel_scalar
    auto start = torch::rand({3, 5});
    auto end = torch::rand({3, 5});
    auto scalar = 0.5;
    // expected and actual
    auto scalar_expected = start + scalar * (end - start);
    auto out = torch::lerp(start, end, scalar);
    // compare
    ASSERT_EQ(out.dtype(), scalar_expected.dtype());
    ASSERT_TRUE(out.allclose(scalar_expected));

    // test lerp_kernel_tensor
    auto weight = torch::rand({3, 5});
    // expected and actual
    auto tensor_expected = start + weight * (end - start);
    out = torch::lerp(start, end, weight);
    // compare
    ASSERT_EQ(out.dtype(), tensor_expected.dtype());
    ASSERT_TRUE(out.allclose(tensor_expected));
  }
}

TEST_F(OperationTest, Cross) {
  for (auto i = 0; i < TEST_AMOUNT; i++) {
    // input
    auto a = torch::rand({10, 3});
    auto b = torch::rand({10, 3});
    // expected
    auto exp = torch::empty({10, 3});
    for (int j = 0; j < 10; j++) {
      auto u1 = a[j][0], u2 = a[j][1], u3 = a[j][2];
      auto v1 = b[j][0], v2 = b[j][1], v3 = b[j][2];
      exp[j][0] = u2 * v3 - v2 * u3;
      exp[j][1] = v1 * u3 - u1 * v3;
      exp[j][2] = u1 * v2 - v1 * u2;
    }
    // actual
    auto out = torch::cross(a, b);
    // compare
    ASSERT_EQ(out.dtype(), exp.dtype());
    ASSERT_TRUE(out.allclose(exp));
  }
}
