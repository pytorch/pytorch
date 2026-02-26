#include <gtest/gtest.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <string>

TEST(TestAccessor, HeaderOnlyTensorAccessor) {
  std::vector<int32_t> v = {11, 12, 13, 21, 22, 23};
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};

  auto acc = torch::headeronly::HeaderOnlyTensorAccessor<int32_t, 2>(
      v.data(), sizes.data(), strides.data());
  EXPECT_EQ(acc[0][0], 11);
  EXPECT_EQ(acc[0][1], 12);
  EXPECT_EQ(acc[0][2], 13);
  EXPECT_EQ(acc[1][0], 21);
  EXPECT_EQ(acc[1][1], 22);
  EXPECT_EQ(acc[1][2], 23);
}

TEST(TestAccessor, HeaderOnlyGenericPackedTensorAccessor) {
  std::vector<int32_t> v = {11, 12, 13, 21, 22, 23};
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};

  auto acc =
      torch::headeronly::HeaderOnlyGenericPackedTensorAccessor<int32_t, 2>(
          v.data(), sizes.data(), strides.data());
  EXPECT_EQ(acc[0][0], 11);
  EXPECT_EQ(acc[0][1], 12);
  EXPECT_EQ(acc[0][2], 13);
  EXPECT_EQ(acc[1][0], 21);
  EXPECT_EQ(acc[1][1], 22);
  EXPECT_EQ(acc[1][2], 23);

  auto tacc = acc.transpose(0, 1);
  EXPECT_EQ(tacc[0][0], 11);
  EXPECT_EQ(tacc[0][1], 21);
  EXPECT_EQ(tacc[1][0], 12);
  EXPECT_EQ(tacc[1][1], 22);
  EXPECT_EQ(tacc[2][0], 13);
  EXPECT_EQ(tacc[2][1], 23);

  try {
    acc.transpose(0, 2);
  } catch (const std::exception& e) {
    EXPECT_TRUE(
        std::string(e.what()).find("HeaderOnlyIndexBoundsCheck") !=
        std::string::npos);
  }
}
