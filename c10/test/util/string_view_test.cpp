#include <c10/util/string_view.h>

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <unordered_map>

TEST(StringViewTest, TransparentStringHash) {
  std::unordered_map<
      std::string,
      int,
      c10::TransparentStringHash,
      std::equal_to<>>
      values{{"value", 1}};

  const std::string_view key = "value";
  EXPECT_EQ(values.find(key)->second, 1);
  EXPECT_TRUE(values.contains(key));
}
