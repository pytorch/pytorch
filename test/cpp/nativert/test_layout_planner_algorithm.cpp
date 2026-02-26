#include <c10/util/Enumerate.h>
#include <gtest/gtest.h>

#include <torch/nativert/executor/memory/Bump.h>
#include <torch/nativert/executor/memory/DisjointStorageGroups.h>
#include <torch/nativert/executor/memory/GreedyBySize.h>

using namespace ::testing;
using namespace torch::nativert;

std::vector<AllocationSpec> create_test_allocation_specs() {
  std::vector<AllocationSpec> specs;

  const std::vector<std::tuple<size_t, size_t, size_t>> test_cases = {
      {0, 1, 32},
      {1, 4, 28},
      {2, 5, 36},
      {3, 5, 16},
      {4, 5, 8},
      {5, 7, 64},
      {6, 8, 10},
      {7, 8, 40},
  };

  specs.reserve(test_cases.size());
  for (const auto& [l_start, l_end, size] : test_cases) {
    specs.push_back(AllocationSpec{AllocationLifetime(l_start, l_end), size});
  };

  return specs;
}

// figure 6 -- https://arxiv.org/pdf/2001.03288
TEST(LayoutPlannerAlgorithmTests, TestGreedyBySize) {
  auto result = GreedyBySizeAllocationPlanner(create_test_allocation_specs());

  EXPECT_EQ(result.total_size, 124);

  auto& allocations = result.allocations;

  EXPECT_EQ(allocations[0].offset, 0);
  EXPECT_EQ(allocations[1].offset, 32);
  EXPECT_EQ(allocations[2].offset, 64);
  EXPECT_EQ(allocations[3].offset, 100);
  EXPECT_EQ(allocations[4].offset, 116);
  EXPECT_EQ(allocations[5].offset, 0);
  EXPECT_EQ(allocations[6].offset, 104);
  EXPECT_EQ(allocations[7].offset, 64);
}

TEST(LayoutPlannerAlgorithmTests, TestBump) {
  auto specs = create_test_allocation_specs();
  auto result = BumpAllocationPlanner(create_test_allocation_specs());

  auto& allocations = result.allocations;

  size_t offset = 0;
  for (auto&& [i, spec] : c10::enumerate(specs)) {
    EXPECT_EQ(allocations[i].offset, offset);
    offset += spec.size;
  }

  EXPECT_EQ(result.total_size, offset);
}

TEST(LayoutPlannerAlgorithmTests, TestStorageGroup) {
  auto specs = create_test_allocation_specs();
  auto result = DisjointStorageGroupsPlanner(create_test_allocation_specs());

  auto& allocations = result.allocations;

  EXPECT_EQ(allocations[0].offset, 0);
  EXPECT_EQ(allocations[1].offset, 36);
  EXPECT_EQ(allocations[2].offset, 0);
  EXPECT_EQ(allocations[3].offset, 100);
  EXPECT_EQ(allocations[4].offset, 140);
  EXPECT_EQ(allocations[5].offset, 36);
  EXPECT_EQ(allocations[6].offset, 140);
  EXPECT_EQ(allocations[7].offset, 100);

  for (auto&& [i, spec] : c10::enumerate(specs)) {
    EXPECT_EQ(allocations[i].size, spec.size);
  }

  EXPECT_EQ(result.total_size, 150);
}
