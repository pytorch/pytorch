#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;

#define ASSERT_ALLCLOSE_TOLERANCES(t1, t2, atol, rtol) \
  ASSERT_TRUE(t1.is_same_size(t2));                    \
  ASSERT_TRUE(t1.allclose(t2, atol, rtol));

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
  const auto input_shape = input.sizes();
  const auto input_ndim = input.dim();

  double eps = 1e-05;
  for (bool undef_weight: {true, false}) {
    for (int normalized_size: {2, 3}) {
      Tensor undef;
      std::vector<int64_t> normalized_shape(normalized_size, 10);
      const auto weight = rand(normalized_shape);
      const auto bias = rand(normalized_shape);
      const int normalized_ndim = normalized_shape.size();
      const int axis = input_ndim - normalized_ndim;
      auto M = prod_intlist(input_shape.cbegin(), input_shape.cbegin()+ axis);
      auto N = prod_intlist(input_shape.cbegin() + axis, input_shape.cend());

      auto out = at::native_layer_norm(
            input, undef_weight ? undef : weight, undef_weight ? undef : bias ,
            M, N, eps);
      auto math_out = at::native::math_native_layer_norm(
            input, undef_weight ? undef : weight, undef_weight ? undef : bias,
            M, N, eps);
      ASSERT_ALLCLOSE_TOLERANCES(std::get<0>(out), std::get<0>(math_out), 1e-4, 1e-6);
      ASSERT_ALLCLOSE_TOLERANCES(std::get<1>(out), std::get<1>(math_out), 1e-4, 1e-6);
      ASSERT_ALLCLOSE_TOLERANCES(std::get<2>(out), std::get<2>(math_out), 1e-4, 1e-6);
    }
  }
}

