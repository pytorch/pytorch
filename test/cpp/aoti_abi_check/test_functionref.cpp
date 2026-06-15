#include <gtest/gtest.h>

#include <torch/headeronly/util/FunctionRef.h>

namespace {
int triple(int x) {
  return x * 3;
}
} // namespace

TEST(TestFunctionRef, TestFunctionRef) {
  torch::headeronly::function_ref<int(int)> ref = triple;
  EXPECT_EQ(ref(4), 12);
  EXPECT_TRUE(static_cast<bool>(ref));

  auto lam = [](int x) { return x + 1; };
  c10::function_ref<int(int)> ref2 = lam;
  EXPECT_EQ(ref2(4), 5);

  torch::headeronly::function_ref<int(int)> empty;
  EXPECT_FALSE(static_cast<bool>(empty));
}
