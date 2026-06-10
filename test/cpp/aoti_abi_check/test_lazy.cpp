#include <gtest/gtest.h>

#include <torch/headeronly/util/Lazy.h>

TEST(TestLazy, TestOptimisticLazy) {
  torch::headeronly::OptimisticLazy<int> lazy;
  int calls = 0;
  int& v = lazy.ensure([&] {
    ++calls;
    return 5;
  });
  EXPECT_EQ(v, 5);
  // Already computed: factory not invoked again.
  lazy.ensure([&] {
    ++calls;
    return 9;
  });
  EXPECT_EQ(calls, 1);
}

namespace {
struct MyLazy : torch::headeronly::OptimisticLazyValue<int> {
  int compute() const override {
    return 11;
  }
};
} // namespace

TEST(TestLazy, TestLazyValue) {
  MyLazy l;
  const torch::headeronly::LazyValue<int>& base = l;
  EXPECT_EQ(base.get(), 11);

  // c10 alias
  c10::PrecomputedLazyValue<int> p(3);
  EXPECT_EQ(p.get(), 3);
}
