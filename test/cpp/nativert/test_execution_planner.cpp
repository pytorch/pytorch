#include <gtest/gtest.h>
#include <torch/nativert/executor/ExecutionPlanner.h>

namespace torch::nativert {

TEST(ExecutionPlannerTest, CreatePlan) {
  auto graph = stringToGraph(R"(
    graph(%x, %y):
  %a = foo(a=%x, b=%y)
  %b = foo1(a=%x, b=%y)
  %c = foo2(c=%a, d=%b)
  return(%c)
  )");

  {
    auto plan = ExecutionPlanner{*graph}.createPlan();

    auto& values_to_free = plan->valuesToFree;
    EXPECT_EQ(values_to_free.size(), 5);

    for (const auto i : c10::irange(3)) {
      EXPECT_TRUE(values_to_free[i].empty());
    }

    EXPECT_EQ(values_to_free[3].size(), 2);
    std::set<int64_t> ids{values_to_free[3].begin(), values_to_free[3].end()};
    EXPECT_EQ(
        ids,
        std::set<int64_t>(
            {graph->tryGetValue("a")->id(), graph->tryGetValue("b")->id()}));

    EXPECT_EQ(values_to_free[4].size(), 0);
  }

  {
    auto static_values = ExecutionPlanner::staticValues(*graph);
    std::set<int64_t> static_ids{static_values.begin(), static_values.end()};
    EXPECT_EQ(
        static_ids,
        std::set<int64_t>(
            {graph->tryGetValue("x")->id(),
             graph->tryGetValue("y")->id(),
             graph->tryGetValue("c")->id()}));
  }
}

} // namespace torch::nativert
