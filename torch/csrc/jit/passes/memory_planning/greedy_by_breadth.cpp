#include <torch/csrc/jit/passes/memory_planning/greedy_by_breadth.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_util.h>

namespace torch {
namespace jit {

std::unordered_map<const Value*, uint64_t> getOpNodeTensorSizes(
    std::vector<const Node*> nodes) {
  std::unordered_map<const Value*, uint64_t> all_tensor_sizes;
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

std::unordered_map<LiveRange, Region, live_range_hash> greedyByOperatorBreadth(
    std::unordered_map<const Value*, uint64_t> managed_tensor_sizes,
    LiveRangesMap live_ranges,
    std::vector<const Node*> ops) {
  auto all_tensor_sizes = getOpNodeTensorSizes(ops);
  std::sort(
      ops.begin(),
      ops.end(),
      [&all_tensor_sizes](const Node* op1, const Node* op2) {
        std::unordered_map<const Node*, uint64_t> breadth_op = {
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
        return breadth_op[op1] >= breadth_op[op2];
      });

  std::unordered_map<LiveRange, uint64_t, live_range_hash> managed_live_ranges;
  for (const auto& item : managed_tensor_sizes) {
    managed_live_ranges[live_ranges[item.first]] = item.second;
  }

  auto cmp = [&](auto v1, auto v2) {
    return managed_tensor_sizes[v1] >= managed_tensor_sizes[v2];
  };
  std::map<const Value*, LiveRange, decltype(cmp)> sorted_size_live_ranges_map(
      cmp);
  std::multiset<std::pair<LiveRange, Region>, _region_offset_cmp>
      ordered_allocations;
  std::unordered_map<LiveRange, Region, live_range_hash> allocations;

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
    ordered_allocations.clear();
    for (const auto& t_val : op_managed_tensors) {
      auto lvr = live_ranges[t_val];
      auto t_size = MemoryPlanner::compute_aligned_tensor_size(
          managed_tensor_sizes[t_val]);
      auto best_offset =
          findOffset(lvr, t_size, managed_live_ranges, ordered_allocations);
      ordered_allocations.insert({lvr, Region{best_offset, t_size}});
    }
    for (auto& item : ordered_allocations) {
      allocations[item.first] = item.second;
    }
  }

  return allocations;
};

} // namespace jit
} // namespace torch