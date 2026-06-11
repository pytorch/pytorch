#include <gtest/gtest.h>

#include <torch/headeronly/util/safe_numerics.h>

#include <cstdint>
#include <vector>

TEST(TestSafeNumerics, TestSafeNumerics) {
  int32_t out = 0;
  EXPECT_FALSE(torch::headeronly::add_overflows<int32_t>(1, 2, &out));
  EXPECT_EQ(out, 3);
  EXPECT_TRUE(torch::headeronly::add_overflows<int32_t>(INT32_MAX, 1, &out));

  int32_t mout = 0;
  EXPECT_FALSE(torch::headeronly::mul_overflows<int32_t>(3, 4, &mout));
  EXPECT_EQ(mout, 12);
  EXPECT_TRUE(c10::mul_overflows<int32_t>(INT32_MAX, 2, &mout));

  std::vector<uint64_t> v{2, 3, 4};
  uint64_t prod = 0;
  EXPECT_FALSE(c10::safe_multiplies_u64(v, &prod));
  EXPECT_EQ(prod, 24u);
}
