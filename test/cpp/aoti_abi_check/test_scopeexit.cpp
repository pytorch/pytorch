#include <gtest/gtest.h>

#include <torch/headeronly/util/ScopeExit.h>

namespace {
struct Inc {
  int* p;
  void operator()() const {
    ++*p;
  }
};
} // namespace

TEST(TestScopeExit, TestScopeExit) {
  int counter = 0;
  {
    torch::headeronly::scope_exit<Inc> guard{Inc{&counter}};
    EXPECT_EQ(counter, 0);
  }
  EXPECT_EQ(counter, 1);

  {
    auto guard = torch::headeronly::make_scope_exit([&] { ++counter; });
    EXPECT_EQ(counter, 1);
  }
  EXPECT_EQ(counter, 2);

  // release() disengages the guard so the callable does not run.
  {
    auto guard = c10::make_scope_exit([&] { ++counter; });
    guard.release();
  }
  EXPECT_EQ(counter, 2);
}
