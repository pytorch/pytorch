// Copyright 2004-present Facebook. All Rights Reserved.

#include <c10/util/accumulate.h>

#include <gtest/gtest.h>

#include <list>
#include <vector>

using namespace ::testing;

TEST(accumulateTest, vector_test) {
  std::vector<int> ints = {1, 2, 3, 4, 5};

  EXPECT_EQ(c10::sum_integers(ints), 1 + 2 + 3 + 4 + 5);
  EXPECT_EQ(c10::multiply_integers(ints), 1 * 2 * 3 * 4 * 5);

  EXPECT_EQ(c10::sum_integers(ints.begin(), ints.end()), 1 + 2 + 3 + 4 + 5);
  EXPECT_EQ(
      c10::multiply_integers(ints.begin(), ints.end()), 1 * 2 * 3 * 4 * 5);

  EXPECT_EQ(c10::sum_integers(ints.begin() + 1, ints.end() - 1), 2 + 3 + 4);
  EXPECT_EQ(
      c10::multiply_integers(ints.begin() + 1, ints.end() - 1), 2 * 3 * 4);

  EXPECT_EQ(c10::numelements_from_dim(2, ints), 3 * 4 * 5);
  EXPECT_EQ(c10::numelements_to_dim(3, ints), 1 * 2 * 3);
  EXPECT_EQ(c10::numelements_between_dim(2, 4, ints), 3 * 4);
  EXPECT_EQ(c10::numelements_between_dim(4, 2, ints), 3 * 4);
}

TEST(accumulateTest, list_test) {
  std::list<int> ints = {1, 2, 3, 4, 5};

  EXPECT_EQ(c10::sum_integers(ints), 1 + 2 + 3 + 4 + 5);
  EXPECT_EQ(c10::multiply_integers(ints), 1 * 2 * 3 * 4 * 5);

  EXPECT_EQ(c10::sum_integers(ints.begin(), ints.end()), 1 + 2 + 3 + 4 + 5);
  EXPECT_EQ(
      c10::multiply_integers(ints.begin(), ints.end()), 1 * 2 * 3 * 4 * 5);

  EXPECT_EQ(c10::numelements_from_dim(2, ints), 3 * 4 * 5);
  EXPECT_EQ(c10::numelements_to_dim(3, ints), 1 * 2 * 3);
  EXPECT_EQ(c10::numelements_between_dim(2, 4, ints), 3 * 4);
  EXPECT_EQ(c10::numelements_between_dim(4, 2, ints), 3 * 4);
}

TEST(accumulateTest, base_cases) {
  std::vector<int> ints = {};

  EXPECT_EQ(c10::sum_integers(ints), 0);
  EXPECT_EQ(c10::multiply_integers(ints), 1);
}

TEST(accumulateTest, errors) {
  std::vector<int> ints = {1, 2, 3, 4, 5};

#ifndef NDEBUG
  EXPECT_THROW(c10::numelements_from_dim(-1, ints), c10::Error);
#endif

  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(c10::numelements_to_dim(-1, ints), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(c10::numelements_between_dim(-1, 10, ints), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(c10::numelements_between_dim(10, -1, ints), c10::Error);

  EXPECT_EQ(c10::numelements_from_dim(10, ints), 1);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(c10::numelements_to_dim(10, ints), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(c10::numelements_between_dim(10, 4, ints), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(c10::numelements_between_dim(4, 10, ints), c10::Error);
}
