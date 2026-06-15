#include <gtest/gtest.h>

#include <torch/headeronly/util/Enumerate.h>

#include <vector>

TEST(TestEnumerate, TestEnumerate) {
  std::vector<int> v{10, 20, 30};
  size_t expected_index = 0;
  for (auto&& [index, element] : torch::headeronly::enumerate(v)) {
    EXPECT_EQ(index, expected_index);
    EXPECT_EQ(element, v[expected_index]);
    ++expected_index;
  }
  EXPECT_EQ(expected_index, v.size());

  // c10 alias
  int sum = 0;
  for (auto&& [index, element] : c10::enumerate(v)) {
    sum += static_cast<int>(index) + element;
  }
  EXPECT_EQ(sum, (0 + 10) + (1 + 20) + (2 + 30));
}
