#include <torch/nativert/executor/memory/Bump.h>

namespace torch::nativert {

LayoutPlan BumpAllocationPlanner(
    const std::vector<AllocationSpec>& allocation_specs) {
  LayoutPlan plan;

  auto& allocations = plan.allocations;
  auto& total_size = plan.total_size;

  allocations.reserve(allocation_specs.size());
  for (const auto& spec : allocation_specs) {
    allocations.push_back(Allocation{
        spec.size,
        total_size,
    });
    total_size += spec.size;
  }

  return plan;
}

} // namespace torch::nativert
