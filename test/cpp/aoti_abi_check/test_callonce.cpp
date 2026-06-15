#include <gtest/gtest.h>

#include <torch/headeronly/util/CallOnce.h>

TEST(TestCallOnce, TestCallOnce) {
  torch::headeronly::once_flag flag;
  int counter = 0;
  for (int i = 0; i < 5; ++i) {
    torch::headeronly::call_once(flag, [&] { ++counter; });
  }
  EXPECT_EQ(counter, 1);

  // c10 alias, with forwarded arguments
  c10::once_flag flag2;
  int counter2 = 0;
  c10::call_once(flag2, [&](int v) { counter2 += v; }, 3);
  c10::call_once(flag2, [&](int v) { counter2 += v; }, 3);
  EXPECT_EQ(counter2, 3);
}
