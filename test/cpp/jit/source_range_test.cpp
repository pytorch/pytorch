#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/source_range.h>

using namespace ::testing;
using namespace ::torch::jit;

std::vector<StringCordView> sampleStringCordViews() {
  std::vector<StringCordView> result;

  std::vector<std::shared_ptr<std::string>> strings;
  strings.push_back(std::make_shared<std::string>("hello world"));
  strings.push_back(std::make_shared<std::string>("nihaoma"));

  std::vector<std::string_view> pieces{*strings[0], *strings[1]};

  result.emplace_back(std::move(pieces), std::move(strings));
  pieces = {"hello worldnihaoma"};
  strings.clear();
  result.emplace_back(std::move(pieces), std::move(strings));
  return result;
}

TEST(SourceRangeTest, test_find) {
  for (const auto& view : sampleStringCordViews()) {
    auto x = view.find("rldni", 0);
    EXPECT_EQ(x, 8) << view.str();
    EXPECT_EQ(view.find("ello", 0), 1);
  }
}

TEST(SourceRangeTest, test_substr) {
  for (const auto& view : sampleStringCordViews()) {
    auto x = view.substr(4, 10).str();
    EXPECT_EQ(x, view.str().substr(4, 10));
    EXPECT_EQ(view.substr(0, view.size()).str(), view.str());
    for (const auto start : c10::irange(view.size())) {
      for (const auto size : c10::irange(view.size())) {
        EXPECT_EQ(
            view.substr(start, size).str(), view.str().substr(start, size));
      }
    }
  }
}

TEST(SourceRangeTest, test_iter_simple) {
  for (const auto& view : sampleStringCordViews()) {
    EXPECT_NE(view.begin(), view.end());
    EXPECT_TRUE(view.begin().has_next());
    EXPECT_EQ(view.str(), std::string(view.begin(), view.end()));
  }
}

TEST(SourceRangeTest, test_iter) {
  int idx = 0;
  for (const auto& view : sampleStringCordViews()) {
    auto iter = view.iter_for_pos(5);
    EXPECT_EQ(*iter, ' ');
    if (idx++ == 0) {
      EXPECT_EQ(iter.rest_line(), " world");
    } else {
      EXPECT_EQ(iter.rest_line(), " worldnihaoma");
    }
    EXPECT_EQ(*iter.next_iter(), 'w');
    EXPECT_EQ(iter.pos(), 5);
    iter = view.iter_for_pos(13);
    EXPECT_EQ(iter.pos(), 13);
  }
}

TEST(SourceRangeTest, SimpleString) {
  Source src("hello");
  EXPECT_EQ(src.num_lines(), 1);
  EXPECT_EQ(src.get_line(0), "hello");
  EXPECT_EQ(src.text_str().str(), "hello");
}
