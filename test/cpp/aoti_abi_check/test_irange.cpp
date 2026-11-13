#include <gtest/gtest.h>

#include <torch/headeronly/util/irange.h>

#include <vector>

namespace torch {
namespace aot_inductor {

TEST(TestIrange, TestRange) {
  using torch::headeronly::integer_range;
  using torch::headeronly::irange;

  std::vector<int> test_vec;
  integer_range<int> range = irange(4, 11);
  for (const auto i : range) {
    test_vec.push_back(i);
  }
  const std::vector<int> correct = {{4, 5, 6, 7, 8, 9, 10}};
  EXPECT_EQ(test_vec, correct);
}

TEST(TestIrange, TestEnd) {
  using torch::headeronly::irange;

  std::vector<int> test_vec;
  for (const auto i : irange(5)) {
    test_vec.push_back(i);
  }
  const std::vector<int> correct = {{0, 1, 2, 3, 4}};
  EXPECT_EQ(test_vec, correct);
}

TEST(TestIrange, TestEmptyReverseRange) {
  using torch::headeronly::irange;

  std::vector<int> test_vec;
  for (const auto i : irange(-3)) {
    test_vec.push_back(i);
  }
  EXPECT_TRUE(test_vec.empty());
}

} // namespace aot_inductor
} // namespace torch
