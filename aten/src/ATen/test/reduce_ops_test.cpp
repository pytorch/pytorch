#include <gtest/gtest.h>

#include <torch/types.h>
#include <torch/utils.h>

using namespace at;

TEST(ReduceOpsTest, MaxValuesAndMinValues) {
  const int W = 10;
  const int H = 10;
  if (hasCUDA()) {
    for (const auto dtype : {kHalf, kFloat, kDouble}) {
      auto a = at::rand({H, W}, TensorOptions(kCUDA).dtype(dtype));
      ASSERT_FLOAT_EQ(
        a.amax(c10::IntArrayRef{0, 1}).item<double>(),
        a.max().item<double>()
      );
      ASSERT_FLOAT_EQ(
        a.amin(c10::IntArrayRef{0, 1}).item<double>(),
        a.min().item<double>()
      );
    }
  }
}
