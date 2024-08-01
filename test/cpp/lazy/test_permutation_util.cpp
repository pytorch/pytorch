#include <gtest/gtest.h>

#include <c10/util/Exception.h>
#include <torch/csrc/lazy/core/permutation_util.h>

namespace torch {
namespace lazy {

TEST(PermutationUtilTest, TestInversePermutation) {
  EXPECT_EQ(InversePermutation({0}), std::vector<int64_t>({0}));
  EXPECT_EQ(InversePermutation({0, 1, 2}), std::vector<int64_t>({0, 1, 2}));
  EXPECT_EQ(
      InversePermutation({1, 3, 2, 0}), std::vector<int64_t>({3, 0, 2, 1}));
  // Not a valid permutation
  EXPECT_THROW(InversePermutation({-1}), c10::Error);
  EXPECT_THROW(InversePermutation({1, 1}), c10::Error);
}

TEST(PermutationUtilTest, TestIsPermutation) {
  EXPECT_TRUE(IsPermutation({0}));
  EXPECT_TRUE(IsPermutation({0, 1, 2, 3}));
  EXPECT_FALSE(IsPermutation({-1}));
  EXPECT_FALSE(IsPermutation({5, 3}));
  EXPECT_FALSE(IsPermutation({1, 2, 3}));
}

TEST(PermutationUtilTest, TestPermute) {
  EXPECT_EQ(
      PermuteDimensions({0}, std::vector<int64_t>({224})),
      std::vector<int64_t>({224}));
  EXPECT_EQ(
      PermuteDimensions({1, 2, 0}, std::vector<int64_t>({3, 224, 224})),
      std::vector<int64_t>({224, 224, 3}));
  // Not a valid permutation
  EXPECT_THROW(
      PermuteDimensions({-1}, std::vector<int64_t>({244})), c10::Error);
  EXPECT_THROW(
      PermuteDimensions({3, 2}, std::vector<int64_t>({244})), c10::Error);
  // Permutation size is different from the to-be-permuted vector size
  EXPECT_THROW(
      PermuteDimensions({0, 1}, std::vector<int64_t>({244})), c10::Error);
  EXPECT_THROW(
      PermuteDimensions({0}, std::vector<int64_t>({3, 244, 244})), c10::Error);
}

} // namespace lazy
} // namespace torch
