#include <gtest/gtest.h>

#include <torch/headeronly/util/string_utils.h>

#include <string>

TEST(TestStringUtils, TestStringUtils) {
  EXPECT_EQ(torch::headeronly::stoi("42"), 42);
  EXPECT_DOUBLE_EQ(torch::headeronly::stod("1.5"), 1.5);
  EXPECT_EQ(torch::headeronly::stoll("100"), 100LL);
  EXPECT_EQ(torch::headeronly::stoull("100"), 100ULL);
  EXPECT_EQ(torch::headeronly::to_string(7), "7");

  // c10 alias
  EXPECT_EQ(c10::stoi("9"), 9);
}
