#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_by_breadth.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>
#include "greedy_util.h"

namespace torch {
namespace jit {

std::vector<MemAllocation> greedyByOperatorBreadth(
    const LivenessMap& liveness_map,
    const FastMap<const Value*, std::pair<UniqueLiveRange, size_t>>&
        managed_values) {
  auto ulvr_cmp = liveRangeStartCmp();

  // sort values by their size (breaking ties according to live range)
  auto val_cmp = [&managed_values, &ulvr_cmp](
                     const Value* v1, const Value* v2) {
    auto item1 = managed_values.at(v1);
    auto item2 = managed_values.at(v2);
    return item1.second == item2.second ? ulvr_cmp(item1.first, item2.first)
                                        : item1.second > item2.second;
  };

  // collect all nodes and their input/output values
  using OpBreadth = std::pair<const Node*, std::vector<const Value*>>;
  std::vector<OpBreadth> ops;

  // this is a hack (using debugNames) but nodes.size() doesn't exist
  ops.reserve(managed_values.begin()
                  ->first->node()
                  ->owningGraph()
                  ->debugNames()
                  .size());
  for (const auto& item : managed_values) {
    auto node = item.first->node();
    std::vector<const Value*> vs;
    std::copy_if(
        node->inputs().begin(),
        node->inputs().end(),
        std::back_inserter(vs),
        [&managed_values](auto inp) { return managed_values.count(inp); });
    std::copy_if(
        node->outputs().begin(),
        node->outputs().end(),
        std::back_inserter(vs),
        [&managed_values](auto outp) { return managed_values.count(outp); });
    std::sort(vs.begin(), vs.end(), val_cmp);
    ops.emplace_back(node, vs);
  }

  // sort all nodes by sum of tensor sizes (i.e., op breadth)
  FastMap<const Node*, size_t> op_breadths;
  for (const auto& item : ops) {
    op_breadths[item.first] = std::accumulate(
        item.second.begin(),
        item.second.end(),
        0,
        [&managed_values](auto acc, auto v) {
          return acc + managed_values.at(v).second;
        });
  }
  auto breadth_cmp = [&op_breadths](OpBreadth item1, OpBreadth item2) {
    return op_breadths[item1.first] > op_breadths[item2.first];
  };
  std::sort(ops.begin(), ops.end(), breadth_cmp);

  // will be ordered by offset (maintained by makeAllocation)
  std::unordered_map<const Value*, MemAllocation> current_allocations;
  current_allocations.reserve(managed_values.size());
  for (const auto& op : ops) {
    for (const auto& t_val : op.second) {
      if (current_allocations.count(t_val)) {
        continue;
      }
      auto ulvr = managed_values.at(t_val).first;
      auto size = managed_values.at(t_val).second;
      makeAllocation(
          ulvr,
          size,
          current_allocations,
          liveness_map,
          GAP_PRIORITY::SMALLEST);
    }
  }

  return orderAllocations(current_allocations);
}

} // namespace jit
} // namespace torch