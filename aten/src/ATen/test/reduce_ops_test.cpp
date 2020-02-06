#include <gtest/gtest.h>

#include <torch/types.h>
#include <torch/utils.h>

using namespace at;

TEST(ReduceOpsTest, MaxValuesAndMinValues) {
  const int W = 10;
  const int H = 10;

  for (const auto dev : {kCPU, kCUDA}) {
    std::vector<at::ScalarType> dtypes = {kFloat, kDouble, kLong};
    if (hasCUDA()) {
      dtypes.push_back(kHalf);
      dtypes.push_back(kShort);
      dtypes.push_back(kInt);
    } else if (dev == kCUDA) {
      // Skip CUDA test in non cuda env
      continue;
    }

    for (const auto dtype : dtypes) {
      auto a = at::rand({H, W}, TensorOptions(dev).dtype(dtype));
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
