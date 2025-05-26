#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>
#include <c10/util/irange.h>

using namespace at;

bool allClose(const at::Tensor& t1, const at::Tensor& t2, double rtol=1e-5, double atol=1e-8) {
  if (!t1.is_same_size(t2)) {
    std::cerr << "Difference in tensor shapes: "
      << t1.sizes() << " v.s. " << t2.sizes() << std::endl;
    return false;
  }
  bool equal = t1.allclose(t2, rtol, atol);
  if (!equal) {
    std::cerr << "Difference in tensor value: \nFirst tensor:\n"
        << t1 << "\nSecond tensor:\n" << t2 << std::endl;
  }
  return equal;
}

#define ASSERT_ALLCLOSE_TOLERANCES(t1, t2, rtol, atol) \
  ASSERT_TRUE(allClose(t1, t2, rtol, atol));

// Ideally we want to test both forward and backward on math kernels but I
// haven't found an easy way to do it.  Currently we only test forward here
// and rely on backward tests of each at:: function used in math kernels.
TEST(MathKernelTest, NativeGroupNorm) {
  int num_channels = 6;
  int N = 2;
  int H = 2, W = 2;
  int HxW = H * W;

  const auto input = randn({N, num_channels, H, W});
  const auto weight = randn({num_channels});
  const auto bias = randn({num_channels});
  double eps = 1e-05;
  for (bool undef_weight: {true, false}) {
    for (int num_groups: {3, 6, 1}) {
      Tensor undef;
      auto out = at::native::native_group_norm(
            input, undef_weight ? undef : weight, undef_weight ? undef : bias,
            N, num_channels, HxW, num_groups, eps);
      auto math_out = at::native::math_group_norm(
            input, undef_weight ? undef : weight, undef_weight ? undef : bias,
            N, num_channels, HxW, num_groups, eps);
      ASSERT_ALLCLOSE_TOLERANCES(std::get<0>(out), std::get<0>(math_out), 1e-4, 1e-6);
      ASSERT_ALLCLOSE_TOLERANCES(std::get<1>(out), std::get<1>(math_out), 1e-4, 1e-6);
      ASSERT_ALLCLOSE_TOLERANCES(std::get<2>(out), std::get<2>(math_out), 1e-4, 1e-6);
    }
  }
}

TEST(MathKernelTest, NativeLayerNorm) {
  const auto input = rand({20, 10, 10, 10});

  double eps = 1e-05;
  for (bool undef_weight: {true, false}) {
    for (int normalized_size: {2, 3}) {
      Tensor undef;
      std::vector<int64_t> normalized_shape(normalized_size, 10);
      const auto weight = rand(normalized_shape);
      const auto bias = rand(normalized_shape);

      auto out = at::native_layer_norm(
            input, normalized_shape, undef_weight ? undef : weight, undef_weight ? undef : bias,
            eps);
      auto math_out = at::native::math_native_layer_norm(
            input, normalized_shape, undef_weight ? undef : weight, undef_weight ? undef : bias,
            eps);
      ASSERT_ALLCLOSE_TOLERANCES(std::get<0>(out), std::get<0>(math_out), 1e-3, 1e-5);
      ASSERT_ALLCLOSE_TOLERANCES(std::get<1>(out), std::get<1>(math_out), 1e-3, 1e-5);
      ASSERT_ALLCLOSE_TOLERANCES(std::get<2>(out), std::get<2>(math_out), 1e-3, 1e-5);
    }
  }
}

TEST(MathKernelTest, Addr) {
  const auto vec1 = arange(1., 4.);
  const auto vec2 = arange(1., 3.);
  const auto M = zeros({3, 2});

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  for (float beta: {1., 1.2, 0.}) {
    // nans and infs are not propagated to the output when beta == 0
    if (beta == 0) {
      M[0][0] = std::numeric_limits<float>::infinity();
      M[2][0] = std::numeric_limits<float>::quiet_NaN();
    }
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    for (float alpha: {1., 2., 0.}) {
      auto out = at::native::addr(M, vec1, vec2, beta, alpha);
      auto math_out = at::native::math_addr(M, vec1, vec2, beta, alpha);
      ASSERT_ALLCLOSE_TOLERANCES(out, math_out, 1e-4, 1e-6);
    }
  }
}

TEST(MathKernelTest, SiluBackward) {
  const auto input = rand({20, 10});
  const auto grad_output = rand({20, 10});
  auto out = at::cpu::silu_backward(grad_output, input);
  auto math_out = at::native::math_silu_backward(grad_output, input);
  ASSERT_ALLCLOSE_TOLERANCES(out, math_out, 1e-4, 1e-6);
}

TEST(MathKernelTest, MishBackward) {
  const auto input = rand({20, 10});
  const auto grad_output = rand({20, 10});
  auto out = at::native::mish_backward(grad_output, input);
  auto math_out = at::native::math_mish_backward(grad_output, input);
  ASSERT_ALLCLOSE_TOLERANCES(out, math_out, 1e-4, 1e-6);
}

TEST(MathKernelTest, Bmm)  {
  auto test_bmm = [](int64_t last_dim) {
    auto x = rand({1, 4, 4}, at::kFloat);
    auto y = rand({1, 4, last_dim}, at::kDouble);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    EXPECT_THROW(auto z = at::bmm(x, y), std::exception);
  };

  test_bmm(5);
  test_bmm(1000);
}
