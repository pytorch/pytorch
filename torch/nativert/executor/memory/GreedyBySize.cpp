#include <iomanip>
#include <limits>
#include <optional>

#include <c10/util/Enumerate.h>
#include <c10/util/Logging.h>
#include <c10/util/irange.h>

#include <torch/nativert/executor/memory/GreedyBySize.h>

namespace {

using namespace torch::nativert;

// we need to track the original order in which allocations were made
// since they will be re-sorted between iterations
struct GreedyAllocation : public Allocation {
  explicit GreedyAllocation(
      Allocation allocation,
      size_t allocation_idx,
      size_t input_spec_idx)
      : Allocation(allocation),
        allocation_index(allocation_idx),
        input_spec_index(input_spec_idx) {}
  // we need to maintain the allocation ordering s.t., we can look up
  // previous allocations directly from descending_allocation_specs_
  // even after allocations has been re-sorted, which happens after
  // each allocation is complete.
  //
  // i.e., this index represents the index of the spec that was used
  // to create this allocation inside descending_allocation_specs_
  // AFTER the sorting was completed.
  size_t allocation_index{0};
  // index of the spec associated with this allocation
  // in the event that the specs get re-ordered
  // in the process of creating allocations
  // e.g.,
  //              allocation_specs[sX, sY, sZ]
  //                                ^   ^   ^
  //                        values[vX, vY, vZ]
  //
  // means that an allocation created from sY
  // will have an input_spec_index of 1
  //
  // this allows us to return to the original
  // ordering before returning the allocations
  size_t input_spec_index{0};
};

struct AllocationSpecWithIndex {
  const AllocationSpec* spec;
  size_t index;
};

// associate specs with their original (unsorted) index
// and then sort them in descending order by byte size
std::vector<AllocationSpecWithIndex> prepare_allocation_specs(
    const std::vector<AllocationSpec>& allocation_specs) {
  std::vector<AllocationSpecWithIndex> specs;
  specs.reserve(allocation_specs.size());

  for (const auto i : c10::irange(allocation_specs.size())) {
    specs.push_back({&allocation_specs[i], i});
  }

  std::sort(specs.begin(), specs.end(), [](auto& lhs, auto& rhs) {
    return lhs.spec->size > rhs.spec->size;
  });

  return specs;
}

} // namespace

namespace torch::nativert {

// https://arxiv.org/pdf/2001.03288
LayoutPlan GreedyBySizeAllocationPlanner(
    const std::vector<AllocationSpec>& allocation_specs) {
  LayoutPlan plan;

  auto descending_allocation_specs = prepare_allocation_specs(allocation_specs);

  std::vector<GreedyAllocation> allocations;
  allocations.reserve(allocation_specs.size());

  auto get_next_offset = [&](const AllocationSpec& spec) -> size_t {
    size_t prev_offset = 0;
    std::optional<size_t> best_offset = std::nullopt;
    size_t smallest_gap = std::numeric_limits<size_t>::max();

    for (const auto& alloc : allocations) {
      if (auto* allocated_spec =
              descending_allocation_specs.at(alloc.allocation_index).spec;
          allocated_spec->not_overlapping_with(spec)) {
        continue;
      }

      if (alloc.offset > prev_offset) {
        if (size_t gap = alloc.offset - prev_offset;
            gap >= spec.size && gap < smallest_gap) {
          smallest_gap = gap;
          best_offset = prev_offset;
        }
      }

      prev_offset = std::max(prev_offset, alloc.offset + alloc.size);
    }

    return best_offset.value_or(prev_offset);
  };

  size_t total_allocation_size = 0;
  for (const auto&& [allocation_index, spec_with_original_index] :
       c10::enumerate(descending_allocation_specs)) {
    auto& spec = spec_with_original_index.spec;

    auto new_allocation = GreedyAllocation(
        Allocation{spec->size, get_next_offset(*spec)},
        allocation_index,
        spec_with_original_index.index);

    total_allocation_size += new_allocation.size;
    plan.total_size =
        std::max(plan.total_size, new_allocation.offset + new_allocation.size);

    VLOG(1) << "allocation with interval " << spec->lifetime.start << "-->"
            << spec->lifetime.end << " placed at offset "
            << new_allocation.offset;

    // insert new allocation while maintaining relative-offset ordering
    // the algorithm is already quadratic because of get_next_offset
    // so this is negligible

    auto it = std::lower_bound(
        allocations.begin(),
        allocations.end(),
        new_allocation,
        [](auto& lhs, auto& rhs) { return lhs.offset < rhs.offset; });
    allocations.insert(it, new_allocation);
  }

  // sort allocations so their ordering is consistent with the input specs
  std::sort(allocations.begin(), allocations.end(), [](auto& lhs, auto& rhs) {
    return lhs.input_spec_index < rhs.input_spec_index;
  });

  plan.allocations.reserve(allocations.size());
  std::move(
      allocations.begin(),
      allocations.end(),
      std::back_inserter(plan.allocations));

  if (plan.total_size > 0) {
    VLOG(1) << std::fixed << std::setprecision(2)
            << "greedy-by-size bytes saved over strictly increasing: "
            << (1.0 - ((float)plan.total_size / (float)total_allocation_size)) *
            100
            << "% (" << total_allocation_size << " - " << plan.total_size
            << " = " << (total_allocation_size - plan.total_size) << " bytes)";
  }

  return plan;
}

} // namespace torch::nativert
