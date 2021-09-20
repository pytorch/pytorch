#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_by_breadth.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_util.h>

namespace torch {
namespace jit {

std::vector<MemAllocation> greedyByOperatorBreadth(
    FastMap<const Value*, std::pair<UniqueLiveRange, size_t>> managed_values) {
  auto ulvr_cmp = liveRangeStartCmp();

  // sort values by their size (breaking ties according to live range)
  auto val_cmp = [&managed_values, &ulvr_cmp](
                     const Value* v1, const Value* v2) {
    auto item1 = managed_values[v1];
    auto item2 = managed_values[v2];
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
          return acc + managed_values[v].second;
        });
  }
  auto breadth_cmp = [&op_breadths](OpBreadth item1, OpBreadth item2) {
    return op_breadths[item1.first] > op_breadths[item2.first];
  };
  std::sort(ops.begin(), ops.end(), breadth_cmp);

  SortedLiveRangeMap<MemRegion> allocations;
  // will be ordered by offset (maintained by makeAllocation)
  std::vector<MemAllocation> ordered_allocations;
  for (const auto& op : ops) {
    for (const auto& t_val : op.second) {
      auto ulvr = managed_values[t_val].first;
      if (allocations.count(ulvr)) {
        continue;
      }
      auto size = managed_values[t_val].second;
      auto mem_alloc = makeAllocation(
          ulvr, size, ordered_allocations, findOffsetWithSmallestGap);
      if (allocations.count(mem_alloc.ulvr)) {
        // this is probably bad to call in a loop?
        TORCH_INTERNAL_ASSERT(
            allocations[mem_alloc.ulvr] == mem_alloc.reg,
            "overwritten allocation");
      } else {
        allocations[mem_alloc.ulvr] = mem_alloc.reg;
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      allocations.size() == ordered_allocations.size(),
      "ill defined allocation ",
      c10::Join("; ", allocations),
      "\n",
      c10::Join("; ", ordered_allocations));

  std::sort(
      ordered_allocations.begin(),
      ordered_allocations.end(),
      [&ulvr_cmp](auto m1, auto m2) { return ulvr_cmp(m1.ulvr, m2.ulvr); });
  return ordered_allocations;
};

} // namespace jit
} // namespace torch