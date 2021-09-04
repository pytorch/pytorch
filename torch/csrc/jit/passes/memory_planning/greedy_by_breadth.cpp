#include <torch/csrc/jit/passes/memory_planning/greedy_by_breadth.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_util.h>

namespace torch {
namespace jit {

std::unordered_map<const Value*, size_t> getOpNodeTensorSizes(
    std::vector<const Node*> nodes) {
  std::unordered_map<const Value*, size_t> all_tensor_sizes;
  for (auto node : nodes) {
    for (const auto& in_v : node->inputs()) {
      auto size = computeStorageSize(*in_v);
      if (size.has_value() && size.value() > 0) {
        all_tensor_sizes.insert({in_v, size.value()});
      }
    }

    for (const auto& out_v : node->outputs()) {
      auto size = computeStorageSize(*out_v);
      if (size.has_value() && size.value() > 0) {
        all_tensor_sizes.insert({out_v, size.value()});
      }
    }
  }
  return all_tensor_sizes;
}

std::vector<MemAllocation> greedyByOperatorBreadth(
    FastMap<const Value*, std::pair<UniqueLiveRange, size_t>> managed_values,
    std::vector<const Node*> ops) {
  auto all_tensor_sizes = getOpNodeTensorSizes(ops);
  std::stable_sort(
      ops.begin(),
      ops.end(),
      [&all_tensor_sizes](const Node* op1, const Node* op2) {
        if (!op1 || !op2) {
          TORCH_WARN(
              "one of op1, op2 is null:", op1 == nullptr, op2 == nullptr);
          return false;
        }
        std::unordered_map<const Node*, size_t> breadth_op = {
            {op1, 0}, {op2, 0}};
        for (const auto& item : breadth_op) {
          auto op = item.first;
          for (const auto& inp : op->inputs()) {
            if (all_tensor_sizes.count(inp)) {
              breadth_op[op] += all_tensor_sizes[inp];
            }
          }
          for (const auto& outp : op->outputs()) {
            if (all_tensor_sizes.count(outp)) {
              breadth_op[op] += all_tensor_sizes[outp];
            }
          }
        }
        return breadth_op[op1] > breadth_op[op2];
      });

  auto ulvr_cmp = liveRangeStartCmp();
  SortedLiveRangeMap<MemRegion> allocations;
  std::vector<MemAllocation> ordered_allocations;

  auto cmp = [&ulvr_cmp, &managed_values](auto v1, auto v2) {
    return managed_values[v1].second == managed_values[v2].second
        ? ulvr_cmp(managed_values[v1].first, managed_values[v2].first)
        : managed_values[v1].second > managed_values[v2].second;
  };

  for (const auto& op : ops) {
    std::vector<const Value*> op_managed_tensors;
    std::copy_if(
        op->inputs().begin(),
        op->inputs().end(),
        std::back_inserter(op_managed_tensors),
        [&](auto v) { return managed_values.count(v); });
    std::copy_if(
        op->outputs().begin(),
        op->outputs().end(),
        std::back_inserter(op_managed_tensors),
        [&](auto v) { return managed_values.count(v); });

    std::stable_sort(op_managed_tensors.begin(), op_managed_tensors.end(), cmp);
    for (const auto& t_val : op_managed_tensors) {
      auto ulvr = managed_values[t_val].first;
      if (allocations.count(ulvr)) {
        continue;
      }
      auto size = managed_values[t_val].second;
      makeAllocation(
          ulvr, size, ordered_allocations, findOffsetWithSmallestGap);
    }

    for (auto& alloc : ordered_allocations) {
      if (allocations.count(alloc.ulvr)) {
        // this is probably bad to call in a loop?
        TORCH_INTERNAL_ASSERT(
            allocations[alloc.ulvr] == alloc.reg, "overwritten allocation");
      } else {
        allocations[alloc.ulvr] = alloc.reg;
      }
    }
  }
  TORCH_INTERNAL_ASSERT(
      allocations.size() == ordered_allocations.size(),
      "ill defined allocation");

  std::stable_sort(
      ordered_allocations.begin(),
      ordered_allocations.end(),
      [&ulvr_cmp](auto m1, auto m2) { return ulvr_cmp(m1.ulvr, m2.ulvr); });
  return ordered_allocations;
};

} // namespace jit
} // namespace torch