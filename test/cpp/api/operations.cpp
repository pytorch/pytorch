#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>
using namespace torch::nn;
using namespace std;
struct OperationTest : torch::test::SeedingFixture {};

TEST_F(OperationTest, Lerp) {
  for (auto i = 0; i < 10; i++) {
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
