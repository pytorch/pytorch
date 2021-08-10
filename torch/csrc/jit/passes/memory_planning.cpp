#include <torch/csrc/jit/passes/memory_planning.h>

#include <c10/core/ScalarType.h>
#include <jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/ops.h>

namespace torch {
namespace jit {

c10::TensorTypePtr checkIsSupported(const Value& value) {
  auto ttp = value.type()->cast<TensorType>();
  if (!ttp) {
    TORCH_WARN("out isn't a tensortype ", *value.type());
  }
  if (!ttp->scalarType().has_value()) {
    ttp = nullptr;
    TORCH_WARN(
        "This output was profiled but didn't have a scalar type: ",
        *ttp,
        ", ",
        value.debugName());
  }
  if (!ttp->sizes().concrete_sizes().has_value()) {
    ttp = nullptr;
    TORCH_WARN(
        "This output was profiled but doesn't have sizes: ",
        *ttp,
        ", ",
        value.debugName());
  }
  return ttp;
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

c10::optional<size_t> computeStorageSize(const c10::TensorTypePtr& ttp) {
  // TODO: when does this fail? answer: in place mutation
  auto numel = ttp->numel();
  if (!numel.has_value()) {
    return c10::nullopt;
  }
  auto scalar_type = ttp->scalarType();
  if (!scalar_type.has_value()) {
    return c10::nullopt;
  }
  auto element_size = c10::elementSize(scalar_type.value());

  return numel.value() * element_size;
}

typedef struct Region {
  uint64_t offset;
  uint64_t size;
} Region;

bool operator==(const LiveRange& lhs, const LiveRange& rhs) {
  return lhs.begin == rhs.begin && lhs.end == rhs.end;
}
bool operator!=(const LiveRange& lhs, const LiveRange& rhs) {
  return !(lhs == rhs);
}

struct live_range_start_comp {
  std::uint64_t operator()(LiveRange const& range1, LiveRange const& range2)
      const {
    return range1.begin < range2.begin;
  }
};

struct live_range_end_comp {
  std::uint64_t operator()(LiveRange const& range1, LiveRange const& range2)
      const {
    return range1.end < range2.end;
  }
};

struct region_size_cmp {
  std::uint64_t operator()(Region const& reg1, Region const& reg2) const {
    return reg1.size < reg2.size;
  }
};

void coalesce_avail(std::set<Region, region_size_cmp>& avail_regions) {
  std::vector<Region> offset_sorted_avail_regions(
      avail_regions.begin(), avail_regions.end());
  std::sort(
      offset_sorted_avail_regions.begin(),
      offset_sorted_avail_regions.end(),
      [](Region const& reg1, Region const& reg2) {
        return reg1.offset < reg2.offset;
      });

  std::vector<Region> coalesced;
  for (auto& offset_sorted_avail_region : offset_sorted_avail_regions) {
    if (!coalesced.empty() &&
        coalesced.back().offset + coalesced.back().size ==
            offset_sorted_avail_region.offset) {
      coalesced.back().size += offset_sorted_avail_region.size;
    } else {
      coalesced.emplace_back(offset_sorted_avail_region);
    }
  }
  avail_regions.clear();
  for (const auto& reg : coalesced) {
    avail_regions.insert(reg);
  }
};

// https://www.usenix.org/legacy/events/vee05/full_papers/p132-wimmer.pdf
std::map<const Value*, Region> linearScanHeuristicAllocations(
    std::map<const Value*, c10::TensorTypePtr> managed_tensor_values,
    LiveRangesMap live_ranges) {
  // WARNING: duplicate ranges not supported
  std::map<LiveRange, const Value*, live_range_start_comp>
      sorted_start_live_ranges_map;
  for (const auto& item : live_ranges) {
    if (managed_tensor_values.count(item.first)) {
      sorted_start_live_ranges_map.insert({item.second, item.first});
    }
  }
  std::vector<LiveRange> sorted_end_live_ranges;
  sorted_end_live_ranges.reserve(sorted_start_live_ranges_map.size());
  for (const auto& item : sorted_start_live_ranges_map) {
    sorted_end_live_ranges.emplace_back(item.first);
  }

  int curr_end_reg = 0;
  std::set<LiveRange, live_range_end_comp> active;
  std::map<LiveRange, Region, live_range_start_comp> alloced_regions;
  std::map<LiveRange, Region, live_range_start_comp> currently_alloced_regions;
  std::set<Region, region_size_cmp> avail_regions;

  for (auto& curr_live_range : sorted_start_live_ranges_map) {
    auto curr_range = curr_live_range.first;

    // expire old intervals
    for (auto& dead_range : sorted_end_live_ranges) {
      if (dead_range.end >= curr_range.begin) {
        break;
      }
      active.erase(dead_range);
      alloced_regions[dead_range] = currently_alloced_regions[dead_range];
      avail_regions.insert(currently_alloced_regions[dead_range]);
      currently_alloced_regions.erase(dead_range);
    }

    auto curr_size = computeStorageSize(
        managed_tensor_values[sorted_start_live_ranges_map[curr_range]]);
    if (!curr_size.has_value()) {
      continue;
    }
    auto aligned_curr_size =
        MemoryPlanner::compute_aligned_tensor_size(curr_size.value());

    // check avail regions
    const Region* reg = nullptr;
    coalesce_avail(avail_regions);
    for (auto& avail_reg : avail_regions) {
      if (avail_reg.size >= aligned_curr_size) {
        reg = &avail_reg;
        break;
      }
    }

    if (reg != nullptr) {
      avail_regions.erase(*reg);
      currently_alloced_regions[curr_range] = {reg->offset, aligned_curr_size};
      // split region (potentially)
      if (reg->size - aligned_curr_size > 0) {
        avail_regions.insert(
            {reg->offset + aligned_curr_size, reg->size - aligned_curr_size});
      }
    } else {
      // if possible spill smallest farthest out alloc
      const LiveRange* swap_lvr = nullptr;
      if (!active.empty()) {
        for (auto lv = active.end(); *lv != curr_range; lv--) {
          auto alloced_reg = currently_alloced_regions[*lv];
          if (alloced_reg.size >= aligned_curr_size) {
            reg = &alloced_reg;
            swap_lvr = &*lv;
          } else {
            break;
          }
        }
      }

      // swap i.e. put new alloc in old spot and malloc old alloc
      if (reg != nullptr) {
        // put new alloc at base of old region
        currently_alloced_regions[curr_range] = {
            reg->offset, aligned_curr_size};
        // split region (potentially)
        if (reg->size - aligned_curr_size > 0) {
          avail_regions.insert(
              {reg->offset + aligned_curr_size, reg->size - aligned_curr_size});
        }
        auto spill_size = currently_alloced_regions[*swap_lvr].size;
        currently_alloced_regions[*swap_lvr] = {curr_end_reg, spill_size};
        curr_end_reg += spill_size;
      } else {
        // create new alloc
        currently_alloced_regions[curr_range] = {
            curr_end_reg, aligned_curr_size};
        curr_end_reg += aligned_curr_size;
      }
    }

    active.insert(curr_range);
  }

  std::map<const Value*, Region> allocations;
  for (auto& item : alloced_regions) {
    auto lvr = item.first;
    Region reg = item.second;
    allocations[sorted_start_live_ranges_map[lvr]] = reg;
  }
  return allocations;
}

Node* insertAllocStorageNode(
    std::shared_ptr<Graph>& graph,
    uint64_t total_size) {
  auto* storage = graph->create(prim::AllocateStorage, 1);
  storage->i_(attr::total_size, total_size);

  auto deviceType = jit::tensorexpr::pickDeviceType(graph);
  if (deviceType.has_value()) {
    storage->i_(attr::device, static_cast<int8_t>(deviceType.value().type()));
  } else {
    storage->i_(attr::device, static_cast<int8_t>(at::kCPU));
  }
  storage->insertBefore(graph->nodes().front());
  return storage;
}

void insertAllocTensorNodes(
    std::shared_ptr<Graph>& graph,
    Node* storage,
    std::map<const Value*, Region> allocations) {
  uint64_t total_size = storage->i(attr::total_size);
  for (auto& allocation : allocations) {
    // const_cast fishy?
    auto node = const_cast<Node*>(allocation.first->node());
    auto region = allocation.second;

    // the way that this node magically *becomes* the out varaint is simply
    // by add an extra input. this is because op resolution happens
    // at runtime via the op registry (by matching on the schema).
    auto* alloc = graph->create(prim::AllocateTensor, 1);
    node->addInput(alloc->output());
    GRAPH_DEBUG("inserting allocation op for ", node->getOperator().schema());
    alloc->insertBefore(node);
    alloc->addInput(storage->output());

    auto ttp = allocation.first->type()->expect<c10::TensorType>();
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

std::pair<std::vector<const Node*>, std::map<const Value*, c10::TensorTypePtr>>
findManagedValues(
    const std::shared_ptr<Graph>& graph,
    std::unordered_set<const Value*> always_alive_values) {
  std::map<const Value*, c10::TensorTypePtr> managed_tensor_values;
  std::unordered_set<const Value*> leaked_values;
  std::vector<const Node*> out_nodes;

  for (auto node : graph->nodes()) {
    if (!hasOutVariant(node)) {
      continue;
    }
    out_nodes.emplace_back(node);
    for (const auto& out_v : node->outputs()) {
      if (always_alive_values.count(out_v)) {
        continue;
      }
      auto ttp = checkIsSupported(*out_v);
      if (ttp) {
        managed_tensor_values.insert({out_v, ttp});
      } else if (isOptimizableContainerType(node)) {
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
  return std::make_pair(out_nodes, managed_tensor_values);
}

void planMemory(std::shared_ptr<Graph>& graph) {
  // find liveness with aliasing
  AliasDb alias_db(graph);
  auto always_alive = jit::GetAlwaysAliveValues(graph, alias_db);
  auto live_ranges = jit::GetLiveness(graph, always_alive, alias_db).second;

  // find out nodes and values that can be managed
  std::vector<const Node*> out_nodes;
  std::map<const Value*, c10::TensorTypePtr> managed_values;
  std::tie(out_nodes, managed_values) = findManagedValues(graph, always_alive);

  auto allocations =
      linearScanHeuristicAllocations(managed_values, live_ranges);

  uint64_t total_size = 0;
  for (const auto& item : allocations) {
    total_size = std::max(total_size, item.second.offset + item.second.size);
  }
  GRAPH_DEBUG("\ngraph before inserting storage node\n", *graph);

  auto storage_node = insertAllocStorageNode(graph, total_size);
  GRAPH_DEBUG("\ngraph after inserting storage node\n", *graph);

  insertAllocTensorNodes(graph, storage_node, allocations);
  GRAPH_DEBUG("\ngraph after inserting alloc nodes\n", *graph);
}

} // namespace jit
} // namespace torch