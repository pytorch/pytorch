#include <gtest/gtest.h>

#include <torch/types.h>
#include <torch/utils.h>

using namespace at;

TEST(ReduceOpsTest, MaxValuesAndMinValues) {
  const int W = 10;
  const int H = 10;
  if (hasCUDA()) {
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    for (const auto dtype : {kHalf, kFloat, kDouble, kShort, kInt, kLong}) {
      auto a = at::rand({H, W}, TensorOptions(kCUDA).dtype(at::kHalf));
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
