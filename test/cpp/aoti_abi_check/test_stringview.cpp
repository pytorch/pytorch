#include <gtest/gtest.h>

#include <torch/headeronly/util/string_view.h>

#include <unordered_set>

TEST(TestStringView, TestStringView) {
  torch::headeronly::c10_string_view sv("hello world");
  EXPECT_EQ(sv.size(), 11u);

  torch::headeronly::basic_string_view<char> bsv("abc");
  EXPECT_EQ(bsv[0], 'a');
  EXPECT_EQ(bsv.size(), 3u);

  // std::hash specialization works
  std::unordered_set<torch::headeronly::c10_string_view> set;
  set.insert(sv);
  EXPECT_EQ(set.count(sv), 1u);

  // c10 aliases
  c10::c10_string_view csv("xyz");
  EXPECT_EQ(csv.size(), 3u);
  c10::basic_string_view<char> cbsv("q");
  EXPECT_EQ(cbsv.size(), 1u);
}
