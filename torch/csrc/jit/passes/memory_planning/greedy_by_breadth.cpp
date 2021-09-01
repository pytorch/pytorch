#include <torch/csrc/jit/passes/memory_planning/greedy_by_breadth.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_util.h>

namespace torch {
namespace jit {

std::unordered_map<const Value*, int64_t> getOpNodeTensorSizes(
    std::vector<const Node*> nodes) {
  std::unordered_map<const Value*, int64_t> all_tensor_sizes;
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
    std::unordered_map<const Value*, int64_t> managed_tensor_sizes,
    LiveRangesMap live_ranges,
    std::vector<const Node*> ops) {
  auto all_tensor_sizes = getOpNodeTensorSizes(ops);
  std::sort(
      ops.begin(),
      ops.end(),
      [&all_tensor_sizes](const Node* op1, const Node* op2) {
        if (!op1 || !op2) {
          TORCH_WARN(
              "one of op1, op2 is null:", op1 == nullptr, op2 == nullptr);
          return false;
        }
        std::unordered_map<const Node*, int64_t> breadth_op = {
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

  std::unordered_map<LiveRange, int64_t, live_range_hash> managed_live_ranges;
  managed_live_ranges.reserve(managed_tensor_sizes.size());
  for (const auto& item : managed_tensor_sizes) {
    managed_live_ranges[live_ranges[item.first]] = item.second;
  }

  auto lvr_cmp = live_range_start_cmp();
  auto cmp = [&](auto v1, auto v2) {
    return managed_tensor_sizes[v1] == managed_tensor_sizes[v2]
        ? lvr_cmp(live_ranges[v1], live_ranges[v2])
        : managed_tensor_sizes[v1] > managed_tensor_sizes[v2];
  };
  std::unordered_map<LiveRange, Region, live_range_hash> allocations;
  std::vector<MemAllocation> ordered_allocations;

  for (const auto& op : ops) {
    std::vector<const Value*> op_managed_tensors;
    std::copy_if(
        op->inputs().begin(),
        op->inputs().end(),
        std::back_inserter(op_managed_tensors),
        [&](auto v) { return managed_tensor_sizes.count(v); });
    std::copy_if(
        op->outputs().begin(),
        op->outputs().end(),
        std::back_inserter(op_managed_tensors),
        [&](auto v) { return managed_tensor_sizes.count(v); });
    std::sort(op_managed_tensors.begin(), op_managed_tensors.end(), cmp);
    for (const auto& t_val : op_managed_tensors) {
      auto lvr = live_ranges[t_val];
      if (allocations.count(lvr)) {
        continue;
      }
      makeAllocation(
          ordered_allocations,
          managed_live_ranges,
          lvr,
          findOffsetWithSmallestGap);
    }

    for (auto& alloc : ordered_allocations) {
      if (allocations.count(alloc.lvr)) {
        // this is probably bad to call in a loop?
        TORCH_INTERNAL_ASSERT(
            allocations[alloc.lvr] == alloc.reg, "overwritten allocation");
      } else {
        allocations[alloc.lvr] = alloc.reg;
      }
    }
  }
  TORCH_INTERNAL_ASSERT(
      allocations.size() == ordered_allocations.size(),
      "ill defined allocation");

  std::sort(
      ordered_allocations.begin(),
      ordered_allocations.end(),
      [&lvr_cmp](auto m1, auto m2) { return lvr_cmp(m1.lvr, m2.lvr); });
  return ordered_allocations;
};

} // namespace jit
} // namespace torch