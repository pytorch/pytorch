#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/torch.h>

#include <test/cpp/api/support.h>
struct OperationTest : torch::test::SeedingFixture {
 protected:
  void SetUp() override {}

  const int TEST_AMOUNT = 10;
};

TEST_F(OperationTest, Lerp) {
  for ([[maybe_unused]] const auto i : c10::irange(TEST_AMOUNT)) {
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
  for ([[maybe_unused]] const auto i : c10::irange(TEST_AMOUNT)) {
    // input
    auto a = torch::rand({10, 3});
    auto b = torch::rand({10, 3});
    // expected
    auto exp = torch::empty({10, 3});
    for (const auto j : c10::irange(10)) {
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

TEST_F(OperationTest, Linear_out) {
  {
    const auto x = torch::arange(100., 118).resize_({3, 3, 2});
    const auto w = torch::arange(200., 206).resize_({3, 2});
    const auto b = torch::arange(300., 303);
    auto y = torch::empty({3, 3, 3});
    at::linear_out(y, x, w, b);
    const auto y_exp = torch::tensor(
        {{{40601, 41004, 41407}, {41403, 41814, 42225}, {42205, 42624, 43043}},
         {{43007, 43434, 43861}, {43809, 44244, 44679}, {44611, 45054, 45497}},
         {{45413, 45864, 46315}, {46215, 46674, 47133}, {47017, 47484, 47951}}},
        torch::kFloat);
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
  {
    const auto x = torch::arange(100., 118).resize_({3, 3, 2});
    const auto w = torch::arange(200., 206).resize_({3, 2});
    auto y = torch::empty({3, 3, 3});
    at::linear_out(y, x, w);
    ASSERT_EQ(y.ndimension(), 3);
    ASSERT_EQ(y.sizes(), torch::IntArrayRef({3, 3, 3}));
    const auto y_exp = torch::tensor(
        {{{40301, 40703, 41105}, {41103, 41513, 41923}, {41905, 42323, 42741}},
         {{42707, 43133, 43559}, {43509, 43943, 44377}, {44311, 44753, 45195}},
         {{45113, 45563, 46013}, {45915, 46373, 46831}, {46717, 47183, 47649}}},
        torch::kFloat);
    ASSERT_TRUE(torch::allclose(y, y_exp));
  }
}
