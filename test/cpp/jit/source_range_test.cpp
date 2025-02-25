#include <gtest/gtest.h>
#include <torch/csrc/jit/frontend/source_range.h>

using namespace ::testing;
using namespace ::torch::jit;

TEST(SourceRangeTest, test_find) {
  std::vector<std::shared_ptr<std::string>> strings;
  strings.push_back(std::make_shared<std::string>("hello world"));
  strings.push_back(std::make_shared<std::string>("nihaoma"));

  std::vector<std::string_view> pieces{*strings[0], *strings[1]};

  StringCordView view(pieces, strings);

  auto x = view.find("rldni", 0);
  EXPECT_EQ(x, 8);
}

TEST(SourceRangeTest, test_substr) {
  std::vector<std::shared_ptr<std::string>> strings;
  strings.push_back(std::make_shared<std::string>("hello world"));
  strings.push_back(std::make_shared<std::string>("nihaoma"));

  std::vector<std::string_view> pieces{*strings[0], *strings[1]};

  StringCordView view(pieces, strings);

  auto x = view.substr(4, 10).str();
  EXPECT_EQ(x, view.str().substr(4, 10));
  EXPECT_EQ(view.substr(0, view.size()).str(), view.str());
}

TEST(SourceRangeTest, test_iter) {
  std::vector<std::shared_ptr<std::string>> strings;
  strings.push_back(std::make_shared<std::string>("hello world"));
  strings.push_back(std::make_shared<std::string>("nihaoma"));

  std::vector<std::string_view> pieces{*strings[0], *strings[1]};

  StringCordView view(pieces, strings);

  auto iter = view.iter_for_pos(5);
  EXPECT_EQ(*iter, ' ');
  EXPECT_EQ(iter.rest_line(), " world");
  EXPECT_EQ(*iter.next_iter(), 'w');
  EXPECT_EQ(iter.pos(), 5);

  iter = view.iter_for_pos(13);
  EXPECT_EQ(iter.pos(), 13);
}
