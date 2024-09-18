// Copyright 2004-present Facebook. All Rights Reserved.

#include <c10/util/irange.h>

#include <gtest/gtest.h>

using namespace ::testing;

TEST(irangeTest, range_test) {
  std::vector<int> test_vec;
  for (const auto i : c10::irange(4, 11)) {
    test_vec.push_back(i);
  }
  const std::vector<int> correct = {{4, 5, 6, 7, 8, 9, 10}};
  ASSERT_EQ(test_vec, correct);
}

TEST(irangeTest, end_test) {
  std::vector<int> test_vec;
  for (const auto i : c10::irange(5)) {
    test_vec.push_back(i);
  }
  const std::vector<int> correct = {{0, 1, 2, 3, 4}};
  ASSERT_EQ(test_vec, correct);
}

TEST(irangeTest, neg_range_test) {
  std::vector<int> test_vec;
  for (const auto i : c10::irange(-2, 3)) {
    test_vec.push_back(i);
  }
  const std::vector<int> correct = {{-2, -1, 0, 1, 2}};
  ASSERT_EQ(test_vec, correct);
}

TEST(irange, empty_reverse_range_two_inputs) {
  std::vector<int> test_vec;
  for (const auto i : c10::irange(3, -3)) {
    test_vec.push_back(i);
    if (i > 20) { // Cap the number of elements we add if something goes wrong
      break;
    }
  }
  const std::vector<int> correct = {};
  ASSERT_EQ(test_vec, correct);
}

TEST(irange, empty_reverse_range_one_input) {
  std::vector<int> test_vec;
  for (const auto i : c10::irange(-3)) {
    test_vec.push_back(i);
    if (i > 20) { // Cap the number of elements we add if something goes wrong
      break;
    }
  }
  const std::vector<int> correct = {};
  ASSERT_EQ(test_vec, correct);
}
