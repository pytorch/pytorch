#include <torch/csrc/jit/passes/memory_planning.h>

#include <jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/runtime/static/memory_planner.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <limits>

namespace torch {
namespace jit {

bool overlap(size_t a, size_t b, size_t c, size_t d) {
  TORCH_INTERNAL_ASSERT(a <= b);
  TORCH_INTERNAL_ASSERT(c <= d);
  size_t outer_len = std::max(b, d) - std::min(a, c);
  size_t interval_len_1 = (b - a), interval_len_2 = (d - c);

  // overflow checking since we're dealing with size_t arithmetic
  // as of today (09/02/2021) linear address width on x86 is 48bits (256TB)
  // so this is unneccessary but "640kB [isn't] enough for anyone"
  // so this is necessary
  if (!valid_add(interval_len_1, interval_len_2) ||
      !valid_sub(outer_len, interval_len_1 + interval_len_2)) {
    // multipoint overlap (sum areas larger than outer area)
    return true;
  } else {
    return false;
  }
}

// live ranges overlap like closed intervals (i.e. if .end of one is the same as
// .begin of another)
// note that we add 1 to the right endpoint since [a,b] doesn't intersect [c,d]
// if [a,b+1) doesn't intersect [c,d+1) (since all endpoints are integers).
bool overlapLiveRange(
    const UniqueLiveRange& ulvr1,
    const UniqueLiveRange& ulvr2) {
  return overlap(
      ulvr1.lvr.begin, ulvr1.lvr.end + 1, ulvr2.lvr.begin, ulvr2.lvr.end + 1);
}

// since memory address are zero indexed, offset + size ends at the *beginning*
// of the (offset+size)th byte. hence overlap is like open intervals (i.e.
// overlap is more than at endpoints)
bool overlapMemRegion(const MemRegion& reg1, const MemRegion& reg2) {
  return overlap(
      reg1.offset,
      reg1.offset + reg1.size,
      reg2.offset,
      reg2.offset + reg2.size);
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
    auto aligned_size = MemoryPlanner::compute_aligned_tensor_size(size);
    allocations.push_back({ulvr, {offset, aligned_size}});
    offset += aligned_size;
  }
  return allocations;
}

c10::optional<size_t> computeStorageSize(const Value& value) {
  auto node_header = getHeader(value.node());
  auto ttp = value.type()->cast<TensorType>();
  if (!ttp) {
    GRAPH_DEBUG(
        node_header,
        "; ",
        value.debugName(),
        " isn't a tensortype: ",
        *value.type());
    return c10::nullopt;
  }
  if (!ttp->scalarType().has_value()) {
    GRAPH_DEBUG(
        node_header,
        "; ",
        value.debugName(),
        " was profiled but didn't have a scalar type: ",
        *ttp);
    return c10::nullopt;
  }
  if (!ttp->sizes().concrete_sizes().has_value()) {
    GRAPH_DEBUG(
        node_header,
        "; ",
        value.debugName(),
        " was profiled but doesn't have sizes: ",
        *ttp);
    return c10::nullopt;
  }

  auto scalar_type = ttp->scalarType();
  if (!scalar_type.has_value()) {
    GRAPH_DEBUG(
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
    GRAPH_DEBUG(
        node_header, "; ", value.debugName(), " doesn't have numel: ", *ttp);
    return c10::nullopt;
  }

  return numel.value() * element_size;
}

Node* insertSlabNode(
    const std::shared_ptr<Graph>& graph,
    size_t total_size,
    c10::optional<at::Device> device_type = c10::nullopt) {
  auto slab = graph->create(prim::AllocateSlab, 1);
  slab->output()->setType(StorageType::get());
  slab->i_(attr::total_size, (int64_t)total_size);

  if (device_type || (device_type = jit::tensorexpr::pickDeviceType(graph))) {
    slab->i_(attr::device_type, static_cast<int8_t>(device_type->type()));
  } else {
    TORCH_INTERNAL_ASSERT(false, "couldn't find device type");
  }
  slab->insertBefore(graph->nodes().front());
  return slab;
}

void insertAllocNodes(
    const std::shared_ptr<Graph>& graph,
    size_t total_size,
    std::vector<std::pair<const Value*, MemRegion>>& managed_value_allocations,
    c10::optional<at::Device> device_type = c10::nullopt) {
  auto slab = insertSlabNode(graph, total_size, device_type);
  auto release_slab = graph->create(prim::ReleaseSlab, 0);
  release_slab->addInput(slab->output());
  release_slab->insertBefore(graph->return_node());

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
    alloc->addInput(slab->output());

    auto ttp = val->type()->expect<c10::TensorType>();
    alloc->ty_(attr::profiled_type, ttp);
    alloc->i_(attr::size, (int64_t)region.size);
    alloc->i_(attr::offset, (int64_t)region.offset);
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
      if (c10::isSubtypeOfList(
              ArrayRef<Argument>(node_schema_args),
              ArrayRef<Argument>(variant_schema.arguments())
                  // e.g. maybe_out_arg_idx = 3 means first 3 arg types should
                  // satisfy subtype relationship
                  .slice(0, maybe_out_arg_idx.value()),
              nullptr)) {
        return true;
      }
    }
  }
  return false;
}

void printLiveRanges(FastMap<const Value*, LiveRange> live_ranges) {
  std::cout << "\"live_ranges\": {\n";
  for (const auto& item : live_ranges) {
    std::cout << "\"%" << item.first->debugName() << "\":"
              << "(" << item.second.begin << "," << item.second.end << "),\n";
  }
  std::cout << "},\n"
            << "\n";
}

void printAllocations(std::vector<MemAllocation> allocations) {
  std::cout << "\"allocations\": {\n";
  for (const auto& item : allocations) {
    std::cout << "(" << item.ulvr.lvr.begin << "," << item.ulvr.lvr.end << ")"
              << ":"
              << "(" << item.reg.offset << "," << item.reg.size << "),\n";
  }
  std::cout << "},\n"
            << "\n";
}

std::pair<
    FastMap<const Value*, std::pair<UniqueLiveRange, size_t>>,
    FastMap<const Value*, std::pair<UniqueLiveRange, size_t>>>
getManagedAndUnManagedValues(const std::shared_ptr<Graph>& graph, bool frozen) {
  AliasDb alias_db(graph, frozen);
  FastSet<const Value*> always_alive_values =
      jit::GetAlwaysAliveValues(graph, alias_db);
  FastMap<const Value*, LiveRange> live_ranges =
      jit::GetLiveness(graph, always_alive_values, alias_db).second;

  FastMap<const Value*, std::pair<UniqueLiveRange, size_t>> managed_values{};
  FastMap<const Value*, std::pair<UniqueLiveRange, size_t>> unmanaged_values{};
  FastMap<Node*, bool> node_has_out_variant;
  for (auto* node : graph->nodes()) {
        auto has_out = hasOutVariant(node);
    node_has_out_variant.insert({node, has_out});
  }

  for (auto node : graph->nodes()) {
    if (!node_has_out_variant[node]) {
      for (const auto& out_v : node->outputs()) {
        unmanaged_values.insert(
            {out_v,
             {{live_ranges[out_v], out_v->debugName()},
              computeStorageSize(*out_v).value_or(0)}});
      }
      continue;
    }
    for (const auto* out_v : node->outputs()) {
      if (always_alive_values.count(out_v)) {
        continue;
      }
      auto size = computeStorageSize(*out_v);
      // this follows static runtime in not trying to manage container types
      // that take a long time to allocate for (tuples and lists of tensors)
      if (size > 0 && !isOptimizableContainerType(node, node_has_out_variant)) {
        managed_values.insert(
            {out_v, {{live_ranges[out_v], out_v->debugName()}, size.value()}});
      } else {
        unmanaged_values.insert(
            {out_v,
             {{live_ranges[out_v], out_v->debugName()},
              computeStorageSize(*out_v).value_or(0)}});
      }
    }
  }

  for (auto* val : always_alive_values) {
    auto size = computeStorageSize(*val);
    unmanaged_values.insert(
        {val, {{live_ranges[val], val->debugName()}, size.value_or(0)}});
  }

  return std::make_pair(unmanaged_values, managed_values);
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
    GRAPH_DEBUG("total allocation is too big ", total_size);
    return false;
  }

  for (const auto& alloc1 : allocations) {
    for (const auto& alloc2 : allocations) {
      if (alloc1 == alloc2) {
        continue;
      }
      if (overlapAllocs(alloc1, alloc2)) {
        GRAPH_DEBUG("overlapping allocations: ", alloc1, ", ", alloc2);
        return false;
      }
    }
  }

  if (allocations.size() != managed_live_ranges.size()) {
    GRAPH_DEBUG(
        "not the right number of allocations: ",
        c10::Join("; ", allocations),
        "\n",
        c10::Join("; ", managed_live_ranges));
    return false;
  }

  for (const auto& alloc : allocations) {
    if (!valid_add(alloc.reg.offset, alloc.reg.size)) {
      GRAPH_DEBUG(
          "allocation ",
          alloc.reg,
          " beyond int64_t mem limit: ",
          sizeof(int64_t));
      return false;
    }

    if (!valid_sub(total_size, alloc.reg.offset + alloc.reg.size)) {
      // if this overflows then alloc.reg.offset + alloc.reg.size > total_size
      GRAPH_DEBUG(
          "allocation exceeds total size: ", alloc.reg, ", ", total_size);
      return false;
    }

    if (managed_live_ranges.count(alloc.ulvr) == 0 ||
        // leq because word alignment increases requirements
        // (recomputing aligned size is overkill here)
        managed_live_ranges[alloc.ulvr] > alloc.reg.size) {
      GRAPH_DEBUG(
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

std::pair<size_t, FastMap<const Value*, std::pair<UniqueLiveRange, size_t>>>
planMemory(const std::shared_ptr<Graph>& graph, Strategy strat, bool frozen) {
  FastMap<const Value*, std::pair<UniqueLiveRange, size_t>> managed_values,
      unmanaged_values;
  std::tie(unmanaged_values, managed_values) =
      getManagedAndUnManagedValues(graph, frozen);
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
    default:
      return {};
  }

  auto total_size = getTotalAllocationSize(allocations);

  TORCH_INTERNAL_ASSERT(
      validateAllocations(allocations, managed_live_ranges, total_size),
      "invalid allocation",
      strat);

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

  insertAllocNodes(graph, total_size, managed_value_allocations);
  GRAPH_DEBUG("\ngraph after inserting alloc nodes\n", *graph);

  return std::make_pair(total_size, unmanaged_values);
}
} // namespace jit
} // namespace torch
