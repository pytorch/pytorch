#include <gtest/gtest.h>

#include <torch/headeronly/util/Synchronized.h>

TEST(TestSynchronized, TestSynchronized) {
  torch::headeronly::Synchronized<int> s(0);
  s.withLock([](int& v) { v = 42; });
  int got = s.withLock([](int& v) { return v; });
  EXPECT_EQ(got, 42);

  // c10 alias
  c10::Synchronized<int> s2(7);
  EXPECT_EQ(s2.withLock([](int& v) { return v; }), 7);
}
