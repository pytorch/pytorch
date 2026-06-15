#include <gtest/gtest.h>

#include <torch/headeronly/util/string_view.h>

#include <sstream>
#include <string_view>
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

  // starts_with / ends_with free functions
  EXPECT_TRUE(torch::headeronly::starts_with(
      std::string_view("hello world"), std::string_view("hello")));
  EXPECT_TRUE(torch::headeronly::ends_with(
      std::string_view("hello world"), std::string_view("world")));
  EXPECT_FALSE(torch::headeronly::starts_with(
      std::string_view("abc"), std::string_view("xyz")));

  // string_view alias (== std::string_view)
  torch::headeronly::string_view alias = "alias";
  EXPECT_EQ(alias.size(), 5u);

  // swap
  torch::headeronly::c10_string_view a("aa"), b("bbb");
  torch::headeronly::swap(a, b);
  EXPECT_EQ(a.size(), 3u);
  EXPECT_EQ(b.size(), 2u);

  // operator<<
  std::ostringstream os;
  os << torch::headeronly::c10_string_view("xyz");
  EXPECT_EQ(os.str(), "xyz");

  // c10 aliases
  c10::c10_string_view csv("xyz");
  EXPECT_EQ(csv.size(), 3u);
  c10::basic_string_view<char> cbsv("q");
  EXPECT_EQ(cbsv.size(), 1u);
}
