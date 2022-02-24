#include <gtest/gtest.h>
#include <torch/csrc/jit/frontend/source_range.h>

using namespace ::testing;
using namespace ::torch::jit;

TEST(SourceRangeTest, test_find) {
  std::vector<std::shared_ptr<std::string>> strings;
  strings.push_back(std::make_shared<std::string>("hello world"));
  strings.push_back(std::make_shared<std::string>("nihaoma"));

  std::vector<c10::string_view> pieces{*strings[0], *strings[1]};

  StringCordView view(pieces, strings);

  auto x = view.find("rldni", 0);
  EXPECT_EQ(x, 8);
}

TEST(SourceRangeTest, test_substr) {
  std::vector<std::shared_ptr<std::string>> strings;
  strings.push_back(std::make_shared<std::string>("hello world"));
  strings.push_back(std::make_shared<std::string>("nihaoma"));

  std::vector<c10::string_view> pieces{*strings[0], *strings[1]};

  StringCordView view(pieces, strings);

  auto x = view.substr(4, 10).str();
  EXPECT_EQ(x, view.str().substr(4, 10));
  EXPECT_EQ(view.substr(0, view.size()).str(), view.str());
}
