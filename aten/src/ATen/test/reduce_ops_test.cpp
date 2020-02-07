#include <gtest/gtest.h>

#include <torch/types.h>
#include <torch/utils.h>

using namespace at;

TEST(ReduceOpsTest, MaxValuesAndMinValues) {
  const int W = 10;
  const int H = 10;

  for (const auto dev : {kCPU, kCUDA}) {
    std::vector<at::ScalarType> dtypes = {kFloat, kDouble, kShort, kInt, kLong};

    // Skip CUDA test in non cuda env
    if (!hasCUDA() && dev == kCUDA) {
      continue;
    }

    // Enable half float test for CUDA
    if (dev == kCUDA) {
      dtypes.push_back(kHalf);
    }

    for (const auto dtype : dtypes) {
      auto a = at::randint(-10, 10, {H, W}, TensorOptions(dev).dtype(dtype));
      ASSERT_FLOAT_EQ(
        a.max_values(c10::IntArrayRef{0, 1}).item<double>(),
        a.max().item<double>()
      );
      ASSERT_FLOAT_EQ(
        a.min_values(c10::IntArrayRef{0, 1}).item<double>(),
        a.min().item<double>()
      );
      ASSERT_FLOAT_EQ(
        a.neg().max_values(c10::IntArrayRef{0, 1}).item<double>(),
        a.neg().max().item<double>()
      );
      ASSERT_FLOAT_EQ(
        a.neg().min_values(c10::IntArrayRef{0, 1}).item<double>(),
        a.neg().min().item<double>()
      );
    }
  }
}
