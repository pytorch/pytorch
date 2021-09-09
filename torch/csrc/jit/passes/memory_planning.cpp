#include <torch/csrc/jit/passes/memory_planning.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>
#include <torch/csrc/jit/passes/memory_planning/linear_scan.h>

#include <jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <limits>

namespace torch {
namespace jit {

int overlap(size_t a, size_t b, size_t c, size_t d) {
  TORCH_INTERNAL_ASSERT(a <= b);
  TORCH_INTERNAL_ASSERT(c <= d);
  size_t outer = std::max(b, d) - std::min(a, c);
  size_t l1 = (b - a), l2 = (d - c);

  // overflow checking since we're dealing with size_t arithmetic
  // as of today (09/02/2021) linear address width on x86 is 48bits (256TB)
  // so this is unneccessary but "640kB [isn't] enough for anyone"
  // so this is necessary
  if (!valid_add(l1, l2) || !valid_sub(outer, l1 + l2)) {
    // sum areas larger than possible outer area (thus overlap)
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

// live ranges overlap like closed intervals (i.e. if .end of one is the same as
// .begin of another)
bool overlapLiveRange(
    const UniqueLiveRange& ulvr1,
    const UniqueLiveRange& ulvr2) {
  return overlap(
             ulvr1.lvr.begin, ulvr1.lvr.end, ulvr2.lvr.begin, ulvr2.lvr.end) <=
      0;
}

// since memory address are zero indexed, offset + size ends at the *beginning*
// of the (offset+size)th byte. hence overlap is like open intervals (i.e.
// overlap is more than at endpoints)
bool overlapMemRegion(const MemRegion& reg1, const MemRegion& reg2) {
  return overlap(
             reg1.offset,
             reg1.offset + reg1.size,
             reg2.offset,
             reg2.offset + reg2.size) < 0;
}

bool overlapAllocs(const MemAllocation& m1, const MemAllocation& m2) {
  return overlapLiveRange(m1.ulvr, m2.ulvr) && overlapMemRegion(m1.reg, m2.reg);
}

// stack all tensors end to end "as you see them" over the entire lifespan of
// the plan
std::vector<MemAllocation> naive(
    SortedLiveRangeMap<size_t> managed_live_ranges) {
  std::vector<MemAllocation> allocations;
  allocations.reserve(managed_live_ranges.size());
  size_t offset = 0;
  for (const auto& item : managed_live_ranges) {
    auto ulvr = item.first;
    auto size = item.second;
    auto id = item.first.id;
    auto aligned_size = MemoryPlanner::computeAlignedTensorSize(size);
    allocations.push_back({ulvr, {offset, aligned_size}});
    offset += aligned_size;
  }
  return allocations;
}

c10::optional<size_t> computeStorageSize(const Value& value) {
  auto node_header = getHeader(value.node());
  auto ttp = value.type()->cast<TensorType>();
  if (!ttp) {
    TORCH_WARN(
        node_header,
        "; ",
        value.debugName(),
        " isn't a tensortype: ",
        *value.type());
    return c10::nullopt;
  }
  if (!ttp->scalarType().has_value()) {
    TORCH_WARN(
        node_header,
        "; ",
        value.debugName(),
        " was profiled but didn't have a scalar type: ",
        *ttp);
    return c10::nullopt;
  }
  if (!ttp->sizes().concrete_sizes().has_value()) {
    TORCH_WARN(
        node_header,
        "; ",
        value.debugName(),
        " was profiled but doesn't have sizes: ",
        *ttp);
    return c10::nullopt;
  }

  auto scalar_type = ttp->scalarType();
  if (!scalar_type.has_value()) {
    TORCH_WARN(
        node_header,
        "; ",
        value.debugName(),
        " doesn't have a scalar type: ",
        *ttp);
    return c10::nullopt;
  }

  auto element_size = c10::elementSize(scalar_type.value());
  // TODO: when does this fail? answer: in place mutation
  auto numel = ttp->numel();
  if (!numel.has_value()) {
    TORCH_WARN(
        node_header, "; ", value.debugName(), " doesn't have numel: ", *ttp);
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
    const std::shared_ptr<Graph>& graph,
    size_t total_size) {
  auto* storage = graph->create(prim::AllocateStorage, 1);
  // there is no e.g. ui_ because pytorch doesn't have a uint64 ScalarType
  storage->i_(attr::total_size, (int64_t)total_size);

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
    const std::shared_ptr<Graph>& graph,
    Node& storage,
    std::vector<std::pair<const Value*, MemRegion>>&
        managed_value_allocations) {
  for (auto& item : managed_value_allocations) {
    auto val = item.first;
    auto region = item.second;

    // const_cast fishy?
    Node* node = const_cast<Node*>(val->node());

    // the way that this node magically *becomes* the out varaint is simply
    // by add an extra input. this is because op resolution happens
    // at runtime via the op registry (by matching on the schema).
    auto* alloc = graph->create(prim::AllocateTensor, 1);
    node->addInput(alloc->output());
    alloc->insertBefore(node);
    alloc->addInput(storage.output());

    auto ttp = val->type()->expect<c10::TensorType>();
    std::vector<int64_t> sizes, strides;
    std::tie(sizes, strides) = getSizesStrides(ttp);

    alloc->i_(attr::size, (int64_t)region.size);
    alloc->i_(attr::offset, (int64_t)region.offset);
    alloc->is_(attr::sizes, sizes);
    alloc->is_(attr::stride, strides);
    alloc->i_(attr::device, static_cast<int8_t>(storage.i(attr::device)));
    alloc->i_(attr::dtype, static_cast<int8_t>(ttp->scalarType().value()));
  }
}

bool hasOutVariant(Node* node) {
  if (!node->maybeSchema()) {
    return false;
  }
  const auto& node_schema_args = node->schema().arguments();
  for (const auto& variant : getAllOperatorsFor(node->kind())) {
    auto variant_schema = variant->schema();
    // a simple check for an out param is insufficient because ops with multiple
    // out variants e.g. aten::cat.out and aten::cat.names_out
    auto maybe_out_arg_idx = variant_schema.argumentIndexWithName("out");
    if (maybe_out_arg_idx) {
      // out arg should be the *next* arg after current args
      if ((size_t)maybe_out_arg_idx.value() != node_schema_args.size()) {
        return false;
      }
      // functions are contravariant in arguments i.e. args should be "broader"
      // in replacement fn i.e. to check variant_schema.args[] <:
      // node_schema.args[] we check that node_schema.args[i] <:
      // variant_schema[i] for all i <= len(node_schema.args)
      c10::isSubtypeOfList(
          ArrayRef<Argument>(node_schema_args),
          ArrayRef<Argument>(variant_schema.arguments())
              // e.g. maybe_out_arg_idx = 3 means first 3 arg types should
              // satisfy subtype relationship
              .slice(0, maybe_out_arg_idx.value()),
          nullptr);
      return true;
    }
  }
  return false;
}

FastMap<const Value*, std::pair<UniqueLiveRange, size_t>> getManagedValues(
    const std::shared_ptr<Graph>& graph,
    bool frozen = false) {
  AliasDb alias_db(graph, frozen);
  FastSet<const Value*> always_alive_values =
      jit::GetAlwaysAliveValues(graph, alias_db);
  FastMap<const Value*, LiveRange> live_ranges =
      jit::GetLiveness(graph, always_alive_values, alias_db).second;

  FastMap<const Value*, std::pair<UniqueLiveRange, size_t>> managed_values;
  FastSet<const Value*> leaked_values;
  FastMap<Node*, bool> node_has_out_variant;
  for (auto* node : graph->nodes()) {
    auto has_out = hasOutVariant(node);
    node_has_out_variant.insert({node, has_out});
  }

  for (auto node : graph->nodes()) {
    if (!node_has_out_variant[node]) {
      continue;
    }
    for (const auto* out_v : node->outputs()) {
      if (always_alive_values.count(out_v)) {
        continue;
      }
      auto size = computeStorageSize(*out_v);
      if (size && size.value() > 0 &&
          !isOptimizableContainerType(node, node_has_out_variant)) {
        managed_values.insert(
            {out_v, {{live_ranges[out_v], out_v->debugName()}, size.value()}});
      } else {
        leaked_values.insert(out_v);
      }
    }
  }

  GRAPH_DEBUG("memory planning leaked values: ", c10::Join(",", leaked_values));
  return managed_values;
}

// "high watermark" of memory
size_t getTotalAllocationSize(std::vector<MemAllocation> allocations) {
  size_t total_size = 0;
  for (const auto& alloc : allocations) {
    total_size = std::max(total_size, alloc.reg.offset + alloc.reg.size);
  }
  return total_size;
}

bool validateAllocations(
    std::vector<MemAllocation> allocations,
    SortedLiveRangeMap<size_t> managed_live_ranges,
    size_t total_size) {
  if (total_size >= (size_t)std::numeric_limits<int64_t>::max()) {
    TORCH_WARN("total allocation is too big ", total_size);
    return false;
  }

  for (const auto& alloc1 : allocations) {
    for (const auto& alloc2 : allocations) {
      if (alloc1 == alloc2) {
        continue;
      }
      if (overlapAllocs(alloc1, alloc2)) {
        TORCH_WARN("overlapping allocations: ", alloc1, ", ", alloc2);
        return false;
      }
    }
  }

  if (allocations.size() != managed_live_ranges.size()) {
    TORCH_WARN(
        "not the right number of allocations: ",
        c10::Join("; ", allocations),
        "\n",
        c10::Join("; ", managed_live_ranges));
    return false;
  }

  for (const auto& alloc : allocations) {
    if (!valid_add(alloc.reg.offset, alloc.reg.size)) {
      TORCH_WARN(
          "allocation ",
          alloc.reg,
          " beyond int64_t mem limit: ",
          sizeof(int64_t));
      return false;
    }

    if (!valid_sub(total_size, alloc.reg.offset + alloc.reg.size)) {
      // if this overflows then alloc.reg.offset + alloc.reg.size > total_size
      TORCH_WARN(
          "allocation exceeds total size: ", alloc.reg, ", ", total_size);
      return false;
    }

    if (managed_live_ranges.count(alloc.ulvr) == 0 ||
        // leq because word alignment increases requirements
        // (recomputing aligned size is overkill here)
        managed_live_ranges[alloc.ulvr] > alloc.reg.size) {
      TORCH_WARN(
          "wrong size allocation: ",
          alloc.ulvr,
          ", ",
          managed_live_ranges[alloc.ulvr],
          ", ",
          alloc.reg.size);
      return false;
    }
  }

  return true;
}

void planMemory(const std::shared_ptr<Graph>& graph, Strategy strat) {
  auto managed_values = getManagedValues(graph);
  SortedLiveRangeMap<size_t> managed_live_ranges;
  for (auto& item : managed_values) {
    auto ulvr = item.second.first;
    auto size = item.second.second;
    managed_live_ranges.insert({ulvr, size});
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
    case Strategy::GREEDY_BY_SIZE_WITH_SMALLEST_GAP: {
      allocations = greedyBySizeWithSmallestGap(managed_live_ranges);
      break;
    }
    case Strategy::GREEDY_BY_SIZE_WITH_FIRST_GAP: {
      allocations = greedyBySizeWithFirstGap(managed_live_ranges);
      break;
    }
    case Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_SMALLEST_GAP: {
      allocations = greedyByLongestAndSizeWithSmallestGap(managed_live_ranges);
      break;
    }
    case Strategy::GREEDY_BY_LONGEST_AND_SIZE_WITH_FIRST_GAP: {
      allocations = greedyByLongestAndSizeWithFirstGap(managed_live_ranges);
      break;
    }
    default:
      return;
  }

  auto total_size = getTotalAllocationSize(allocations);

  TORCH_INTERNAL_ASSERT(
      validateAllocations(allocations, managed_live_ranges, total_size),
      "invalid allocation",
      strat);

  GRAPH_DEBUG("\ngraph before inserting storage node\n", *graph);

  auto storage_node = insertAllocStorageNode(graph, total_size);
  GRAPH_DEBUG("\ngraph after inserting storage node\n", *graph);

  // the only way to identify an allocation with the value that needs it
  // is by matching on live range and size request. but we need the original
  // size request because alignment will grow the request (concretely
  // doing
  // managed_values[alloc.lvr].second == allocations_map[alloc.lvr].size
  // will fail because MemRegion.size >= managed_values[alloc.lvr].second
  SortedLiveRangeMap<MemRegion> allocations_map;
  for (const auto& alloc : allocations) {
    allocations_map.insert({alloc.ulvr, alloc.reg});
  }

  std::vector<std::pair<const Value*, MemRegion>> managed_value_allocations;
  managed_value_allocations.reserve(managed_values.size());
  for (auto& item : managed_values) {
    auto val = item.first;
    auto ulvr = item.second.first;
    auto reg = allocations_map.at(ulvr);
    managed_value_allocations.emplace_back(val, reg);
  }

  insertAllocTensorNodes(graph, *storage_node, managed_value_allocations);
  GRAPH_DEBUG("\ngraph after inserting alloc nodes\n", *graph);
}
} // namespace jit
} // namespace torch
