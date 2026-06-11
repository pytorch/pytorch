#include <gtest/gtest.h>

#include <torch/headeronly/util/flat_hash_map.h>

#include <string>

TEST(TestFlatHashMap, TestFlatHashMap) {
  ska::flat_hash_map<int, std::string> m;
  m[1] = "one";
  m[2] = "two";
  EXPECT_EQ(m.size(), 2u);
  EXPECT_EQ(m.at(1), "one");
  EXPECT_EQ(m.count(3), 0u);
  EXPECT_EQ(m.count(2), 1u);
}
