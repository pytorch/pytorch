#include <gtest/gtest.h>

#include <torch/headeronly/util/irange.h>

#include <vector>

TEST(TestIrange, TestIrange) {
  std::vector<int> seen;
  for (auto i : torch::headeronly::irange(3)) {
    seen.push_back(static_cast<int>(i));
  }
  EXPECT_EQ(seen, (std::vector<int>{0, 1, 2}));

  int count = 0;
  for (auto i : c10::irange(2, 5)) {
    (void)i;
    ++count;
  }
  EXPECT_EQ(count, 3);

  // negative end -> empty range (one-sided)
  int neg = 0;
  for (auto i : torch::headeronly::irange(-3)) {
    (void)i;
    ++neg;
  }
  EXPECT_EQ(neg, 0);

  torch::headeronly::integer_range<int> r(0, 2);
  EXPECT_EQ(*r.begin(), 0);
}
