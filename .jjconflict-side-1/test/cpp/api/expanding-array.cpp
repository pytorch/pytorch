#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/torch.h>

#include <test/cpp/api/support.h>

#include <cstddef>
#include <initializer_list>
#include <vector>

struct ExpandingArrayTest : torch::test::SeedingFixture {};

TEST_F(ExpandingArrayTest, CanConstructFromInitializerList) {
  torch::ExpandingArray<5> e({1, 2, 3, 4, 5});
  ASSERT_EQ(e.size(), 5);
  for (const auto i : c10::irange(e.size())) {
    ASSERT_EQ((*e)[i], i + 1);
  }
}

TEST_F(ExpandingArrayTest, CanConstructFromVector) {
  torch::ExpandingArray<5> e(std::vector<int64_t>{1, 2, 3, 4, 5});
  ASSERT_EQ(e.size(), 5);
  for (const auto i : c10::irange(e.size())) {
    ASSERT_EQ((*e)[i], i + 1);
  }
}

TEST_F(ExpandingArrayTest, CanConstructFromArray) {
  torch::ExpandingArray<5> e(std::array<int64_t, 5>({1, 2, 3, 4, 5}));
  ASSERT_EQ(e.size(), 5);
  for (const auto i : c10::irange(e.size())) {
    ASSERT_EQ((*e)[i], i + 1);
  }
}

TEST_F(ExpandingArrayTest, CanConstructFromSingleValue) {
  torch::ExpandingArray<5> e(5);
  ASSERT_EQ(e.size(), 5);
  for (const auto i : c10::irange(e.size())) {
    ASSERT_EQ((*e)[i], 5);
  }
}

TEST_F(
    ExpandingArrayTest,
    ThrowsWhenConstructedWithIncorrectNumberOfArgumentsInInitializerList) {
  ASSERT_THROWS_WITH(
      torch::ExpandingArray<5>({1, 2, 3, 4, 5, 6, 7}),
      "Expected 5 values, but instead got 7");
}

TEST_F(
    ExpandingArrayTest,
    ThrowsWhenConstructedWithIncorrectNumberOfArgumentsInVector) {
  ASSERT_THROWS_WITH(
      torch::ExpandingArray<5>(std::vector<int64_t>({1, 2, 3, 4, 5, 6, 7})),
      "Expected 5 values, but instead got 7");
}
