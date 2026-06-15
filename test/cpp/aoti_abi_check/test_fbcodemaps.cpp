#include <gtest/gtest.h>

#include <torch/headeronly/util/FbcodeMaps.h>

#include <string>

TEST(TestFbcodeMaps, TestFbcodeMaps) {
  torch::headeronly::FastMap<int, std::string> m;
  m[1] = "one";
  EXPECT_EQ(m.at(1), "one");

  torch::headeronly::FastSet<int> s;
  s.insert(2);
  s.insert(2);
  EXPECT_EQ(s.size(), 1u);

  // c10 alias
  c10::FastMap<int, int> m2;
  m2[5] = 6;
  EXPECT_EQ(m2.at(5), 6);
  c10::FastSet<int> s2;
  s2.insert(9);
  EXPECT_EQ(s2.count(9), 1u);
}
