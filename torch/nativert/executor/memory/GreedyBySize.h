#pragma once

#include <torch/nativert/executor/memory/LayoutPlannerAlgorithm.h>

namespace torch::nativert {

LayoutPlan GreedyBySizeAllocationPlanner(
    const std::vector<AllocationSpec>& allocation_specs);

} // namespace torch::nativert
