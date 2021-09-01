#include <torch/csrc/jit/passes/memory_planning.h>
#include <torch/csrc/jit/passes/memory_planning/linear_scan.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_by_breadth.h>

#include <jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/static/ops.h>

namespace torch {
namespace jit {

int intersectArea(int64_t a, int64_t b, int64_t c, int64_t d) {
  TORCH_INTERNAL_ASSERT(a <= b);
  TORCH_INTERNAL_ASSERT(c <= d);
  int64_t outer = std::max(b, d) - std::min(a, c);
  int64_t l1 = (b - a), l2 = (d - c);
  int64_t test = 0;
  if (__builtin_saddll_overflow(l1, l2, &test)) {
    // sum areas larger than possible outer area (thus overlap)
    return -1;
  } else if (__builtin_ssubll_overflow(outer, l1 + l2, &test)) {
    // multipoint overlap (sum areas larger than outer area)
    return -1;
  } else if (outer - (l1 + l2) > 0) {
    // outer area larger than sum (no overlap)
    return 1;
  } else {
    // outer area equals sum area (single point overlap)
    return 0;
  }
}

bool intersectLiveRange(LiveRange lvr1, LiveRange lvr2) {
  return intersectArea(lvr1.begin, lvr1.end, lvr2.begin, lvr2.end) <= 0;
}

bool intersectMemRegion(MemRegion reg1, MemRegion reg2) {
  // greater than 1 point overlap
  return intersectArea(
             reg1.offset,
             reg1.offset + reg1.size,
             reg2.offset,
             reg2.offset + reg2.size) < 0;
}

std::vector<MemAllocation> naive(
    std::unordered_map<LiveRange, int64_t, live_range_hash>
        managed_live_ranges) {
  std::map<LiveRange, int64_t, live_range_start_cmp> sorted_managed_live_ranges(
      managed_live_ranges.begin(), managed_live_ranges.end());
  std::vector<MemAllocation> allocations;
  allocations.reserve(managed_live_ranges.size());
  int64_t offset = 0;
  for (const auto& item : sorted_managed_live_ranges) {
    auto aligned_size = MemoryPlanner::computeAlignedTensorSize(item.second);
    allocations.push_back({item.first, {offset, aligned_size}});
    offset += aligned_size;
  }
  return allocations;
}

c10::optional<int64_t> computeStorageSize(const Value& value) {
  auto ttp = value.type()->cast<TensorType>();
  if (!ttp) {
    TORCH_WARN("out isn't a tensortype ", *value.type());
    return c10::nullopt;
  }
  if (!ttp->scalarType().has_value()) {
    TORCH_WARN(
        "This output was profiled but didn't have a scalar type: ",
        *ttp,
        ", ",
        value.debugName());
    return c10::nullopt;
  }
  if (!ttp->sizes().concrete_sizes().has_value()) {
    TORCH_WARN(
        "This output was profiled but doesn't have sizes: ",
        *ttp,
        ", ",
        value.debugName());
    return c10::nullopt;
  }

  auto scalar_type = ttp->scalarType();
  if (!scalar_type.has_value()) {
    TORCH_WARN(
        "This value doesn't have a scalar type", *ttp, ", ", value.debugName());
    return c10::nullopt;
  }

  auto element_size = c10::elementSize(scalar_type.value());
  // TODO: when does this fail? answer: in place mutation
  auto numel = ttp->numel();
  if (!numel.has_value()) {
    TORCH_WARN("doesn't have numel", *ttp, ", ", value.debugName());
    return c10::nullopt;
  }

  return numel.value() * element_size;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> getSizesStrides(
    const c10::TensorTypePtr& ttp) {
  std::vector<int64_t> sizes;
  auto _sizes = ttp->sizes().concrete_sizes();
  // TODO: why does this break? answer: in place mutation
  // also %9 : Long(requires_grad=0, device=cpu) = prim::Constant[value={0}]()
  if (_sizes.has_value() && _sizes.value().size() > 0 &&
      _sizes.value()[0] != 0) {
    sizes = _sizes.value();
  } else {
    sizes = std::vector<int64_t>{0};
  }
  std::vector<int64_t> strides;
  auto _strides = ttp->strides().concrete_sizes();
  if (_strides.has_value() && _strides.value().size() > 0 &&
      _strides.value()[0] != 0) {
    strides = _strides.value();
  } else {
    strides = at::detail::defaultStrides(sizes);
  }
  return std::make_pair(sizes, strides);
}

Node* insertAllocStorageNode(
    std::shared_ptr<Graph>& graph,
    int64_t total_size) {
  auto* storage = graph->create(prim::AllocateStorage, 1);
  storage->i_(attr::total_size, total_size);

  auto device_type = jit::tensorexpr::pickDeviceType(graph);
  if (device_type.has_value()) {
    storage->i_(attr::device, static_cast<int8_t>(device_type.value().type()));
  } else {
    storage->i_(attr::device, static_cast<int8_t>(at::kCPU));
  }
  storage->insertBefore(graph->nodes().front());
  return storage;
}

void insertAllocTensorNodes(
    std::shared_ptr<Graph>& graph,
    Node* storage,
    std::vector<MemAllocation> allocations,
    std::map<LiveRange, const Value*, live_range_start_cmp>
        managed_range_values) {
  std::unordered_map<LiveRange, MemRegion, live_range_hash> allocations_map;
  allocations_map.reserve(allocations.size());
  for (const auto& item : allocations) {
    allocations_map[item.lvr] = item.reg;
  }

  int64_t total_size = storage->i(attr::total_size);
  for (auto& item : managed_range_values) {
    auto lvr = item.first;
    auto region = allocations_map[lvr];
    auto allocation = item.second;

    // const_cast fishy?
    auto node = const_cast<Node*>(allocation->node());

    // the way that this node magically *becomes* the out varaint is simply
    // by add an extra input. this is because op resolution happens
    // at runtime via the op registry (by matching on the schema).
    auto* alloc = graph->create(prim::AllocateTensor, 1);
    node->addInput(alloc->output());
    GRAPH_DEBUG("inserting allocation op for ", node->getOperator().schema());
    alloc->insertBefore(node);
    alloc->addInput(storage->output());

    auto ttp = allocation->type()->expect<c10::TensorType>();
    std::vector<int64_t> sizes, strides;
    std::tie(sizes, strides) = getSizesStrides(ttp);
    TORCH_CHECK(
        region.offset + region.size <= total_size,
        "trying to create an allocation that exceeds previously planned memory");
    alloc->i_(attr::size, region.size);
    alloc->i_(attr::offset, region.offset);
    alloc->is_(attr::sizes, sizes);
    alloc->is_(attr::stride, strides);
    alloc->i_(attr::device, static_cast<int8_t>(storage->i(attr::device)));
    alloc->i_(attr::dtype, static_cast<int8_t>(ttp->scalarType().value()));
  }
}

bool hasOutVariant(Node* node) {
  for (const auto& variant : getAllOperatorsFor(node->kind())) {
    auto variant_args = variant->schema().arguments();
    /* TODO
      aten::cat.names_out(Tensor[] tensors, str dim, *, Tensor(a!) out) ->
      (Tensor(a!)) aten::cat.out(Tensor[] tensors, int dim=0, *,
      Tensor(a!) out) -> (Tensor(a!))
    */
    auto maybe_out_arg =
        std::find_if(variant_args.begin(), variant_args.end(), [](auto arg) {
          return arg.name() == "out";
        });
    if (maybe_out_arg != variant_args.end()) {
      return true;
    }
  }
  return false;
}

std::pair<std::vector<const Node*>, std::unordered_map<const Value*, int64_t>>
getManagedValues(
    const std::shared_ptr<Graph>& graph,
    std::unordered_set<const Value*> always_alive_values) {
  std::unordered_map<const Value*, int64_t> managed_tensor_values;
  std::unordered_set<const Value*> leaked_values;
  std::vector<const Node*> out_nodes;

  FastMap<Node*, bool> node_has_out_variant;
  for (auto* node : graph->nodes()) {
    node_has_out_variant.insert({node, hasOutVariant(node)});
  }

  for (auto node : graph->nodes()) {
    if (!node_has_out_variant[node]) {
      continue;
    }
    out_nodes.emplace_back(node);
    for (const auto* out_v : node->outputs()) {
      if (always_alive_values.count(out_v)) {
        continue;
      }
      auto size = computeStorageSize(*out_v);
      if (size.has_value() && size.value() > 0) {
        managed_tensor_values.insert({out_v, size.value()});
      } else if (isOptimizableContainerType(node, node_has_out_variant)) {
        leaked_values.insert(out_v);
      } else {
        TORCH_WARN(
            "not handling unsupported value: ",
            out_v->debugName(),
            " ",
            *out_v->type());
        leaked_values.insert(out_v);
      }
    }
  }
  GRAPH_DEBUG("memory planning leaked values: ", c10::Join(",", leaked_values));
  return std::make_pair(out_nodes, managed_tensor_values);
}

std::tuple<
    std::vector<const Node*>,
    std::unordered_map<const Value*, int64_t>,
    std::unordered_map<const Value*, LiveRange>>
getManagedStuff(std::shared_ptr<Graph>& graph) {
  AliasDb alias_db(graph);
  auto always_alive = jit::GetAlwaysAliveValues(graph, alias_db);
  auto live_ranges = jit::GetLiveness(graph, always_alive, alias_db).second;
  std::vector<const Node*> out_nodes;
  std::unordered_map<const Value*, int64_t> managed_tensor_values;
  std::tie(out_nodes, managed_tensor_values) =
      getManagedValues(graph, always_alive);

  std::unordered_map<const Value*, LiveRange> managed_ranges;
  for (const auto& lvr : live_ranges) {
    if (managed_tensor_values.count(lvr.first) > 0) {
      managed_ranges.insert(lvr);
    }
  }
  return std::make_tuple(out_nodes, managed_tensor_values, managed_ranges);
}

int64_t getTotalAllocationSize(std::vector<MemAllocation> allocations) {
  int64_t total_size = 0;
  for (const auto& alloc : allocations) {
    total_size = std::max(total_size, alloc.reg.offset + alloc.reg.size);
  }
  return total_size;
}

bool intersectAllocs(MemAllocation m1, MemAllocation m2) {
  return intersectLiveRange(m1.lvr, m2.lvr) &&
         intersectMemRegion(m1.reg, m2.reg);
}

bool validateAllocations(std::vector<MemAllocation> allocations) {
  for (const auto& alloc1 : allocations) {
    for (const auto& alloc2 : allocations) {
      if (alloc1 == alloc2) {
        continue;
      }
      if (intersectAllocs(alloc1, alloc2)) {
        std::cerr << alloc1 << "," << alloc2 << "\n";
        return false;
      }
    }
  }
  return true;
}

std::ostream& printAllocation(
    std::ostream& out,
    std::vector<MemAllocation> allocations,
    std::map<LiveRange, const Value*, live_range_start_cmp> managed_ranges) {
  std::map<LiveRange, MemRegion, live_range_start_cmp> allocations_map;
  for (const auto& item : allocations) {
    allocations_map[item.lvr] = item.reg;
  }

  for (const auto& item : managed_ranges) {
    auto lvr = item.first;
    auto val = item.second;
    auto alloced_reg = allocations_map[lvr];
    out << val->debugName() << ": " << lvr << " " << alloced_reg << "\n";
  }

  return out;
}

void planMemory(std::shared_ptr<Graph>& graph, Strategy strat) {
  std::unordered_map<const Value*, int64_t> managed_value_sizes;
  std::unordered_map<const Value*, LiveRange> managed_value_ranges;
  std::vector<const Node*> out_nodes;
  std::tie(out_nodes, managed_value_sizes, managed_value_ranges) =
      getManagedStuff(graph);

  std::unordered_map<LiveRange, int64_t, live_range_hash> managed_live_ranges;
  managed_live_ranges.reserve(managed_value_sizes.size());
  for (const auto& item : managed_value_sizes) {
    managed_live_ranges[managed_value_ranges[item.first]] = item.second;
  }
  std::vector<MemAllocation> allocations;

  switch (strat) {
    case Strategy::NAIVE: {
      allocations = naive(managed_live_ranges);
      break;
    }
    case Strategy::LINEAR_SCAN: {
      allocations = linearScanHeuristic(managed_live_ranges);
      break;
    };
    case Strategy::GREEDY_BY_SIZE: {
      allocations = greedyBySize(managed_live_ranges);
      break;
    }
    case Strategy::GREEDY_BY_SIZE_WITH_FIRST_GAP: {
      allocations = greedyBySizeWithFirstGap(managed_live_ranges);
      break;
    }
    case Strategy::GREEDY_BY_LONGEST_AND_SIZE: {
      allocations = greedyBySizeAndLongestWithFirstGap(managed_live_ranges);
      break;
    }
    case Strategy::GREEDY_BY_BREADTH: {
      allocations = greedyByOperatorBreadth(
          managed_value_sizes, managed_value_ranges, out_nodes);
      break;
    }
    default:
      return;
  }

  TORCH_INTERNAL_ASSERT(
      validateAllocations(allocations), "invalid allocation", strat);

  auto total_size = getTotalAllocationSize(allocations);

  std::map<LiveRange, const Value*, live_range_start_cmp> managed_range_values;
  for (const auto& item : managed_value_ranges) {
    if (managed_range_values.count(item.second)) {
      TORCH_WARN(
          "overlapping live ranges ",
          item.first->debugName(),
          " with ",
          managed_range_values.at(item.second)->debugName());
    }
    managed_range_values.insert({item.second, item.first});
  }

  std::stringstream allocs_str;
  printAllocation(allocs_str, allocations, managed_range_values);
  GRAPH_DEBUG("\nallocs\n", allocs_str.str());

  GRAPH_DEBUG("\ngraph before inserting storage node\n", *graph);

  auto storage_node = insertAllocStorageNode(graph, total_size);
  GRAPH_DEBUG("\ngraph after inserting storage node\n", *graph);

  insertAllocTensorNodes(
      graph, storage_node, allocations, managed_range_values);
  GRAPH_DEBUG("\ngraph after inserting alloc nodes\n", *graph);
}
} // namespace jit
} // namespace torch
