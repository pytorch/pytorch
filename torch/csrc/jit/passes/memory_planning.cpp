#include <torch/csrc/jit/passes/memory_planning.h>
#include <torch/csrc/jit/passes/memory_planning/MemoryPlanningAllocator.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_by_breadth.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>
#include <torch/csrc/jit/passes/memory_planning/linear_scan.h>

#include <regex>

#include <c10/util/Backtrace.h>
#include <jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/static/ops.h>
#include <aten/src/ATen/core/interned_strings.h>

namespace torch {
namespace jit {

c10::optional<uint64_t> computeStorageSize(const Value& value) {
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
    std::unordered_map<LiveRange, Region, live_range_hash> allocations,
    std::map<LiveRange, const Value*, live_range_start_cmp>
        manage_range_values) {
  uint64_t total_size = storage->i(attr::total_size);
  for (auto& item : manage_range_values) {
    auto lvr = item.first;
    auto region = allocations[lvr];
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

std::pair<std::vector<const Node*>, std::unordered_map<const Value*, uint64_t>>
getManagedValues(
    const std::shared_ptr<Graph>& graph,
    std::unordered_set<const Value*> always_alive_values) {
  std::unordered_map<const Value*, uint64_t> managed_tensor_values;
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
      auto size = computeStorageSize(*out_v);
      if (size.has_value() && size.value() > 0) {
        managed_tensor_values.insert({out_v, size.value()});
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

std::tuple<
    std::vector<const Node*>,
    std::unordered_map<const Value*, uint64_t>,
    std::unordered_map<const Value*, LiveRange>>
getManagedStuff(std::shared_ptr<Graph>& graph) {
  AliasDb alias_db(graph);
  auto always_alive = jit::GetAlwaysAliveValues(graph, alias_db);
  auto live_ranges = jit::GetLiveness(graph, always_alive, alias_db).second;
  std::vector<const Node*> out_nodes;
  std::unordered_map<const Value*, uint64_t> managed_tensor_values;
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

uint64_t getTotalAllocationSize(
    std::unordered_map<LiveRange, Region, live_range_hash>
        managed_live_ranges) {
  uint64_t total_size = 0;
  for (const auto& item : managed_live_ranges) {
    total_size = std::max(total_size, item.second.offset + item.second.size);
  }
  return total_size;
}

void printAllocation(
    std::unordered_map<LiveRange, Region, live_range_hash> allocations,
    std::map<LiveRange, const Value*, live_range_start_cmp> managed_ranges) {
  for (const auto& item : managed_ranges) {
    auto lvr = item.first;
    auto val = item.second;
    auto alloced_reg = allocations[lvr];
    std::cout << val->debugName() << ": " << lvr << " " << alloced_reg << "\n";
  }
}

c10::Symbol getNodeKindFromBt(std::string bt) {
  auto frame_strs = c10::backTraceToVecStr(bt);

  // e.g. at::native::_convolution(...) + 11551 (0x1275e23bf in
  // libtorch_cpu.dylib) is this mapping at::native::_convolution - >
  // aten::_convolution robust?
  // when does <ident> parens fail?

  std::string native_op_str = "";
  std::regex rgx(R"(^at::native::(\w+)\(.*)");
  std::smatch match;
  auto frame_str = frame_strs.rbegin();
  for (; frame_str != frame_strs.rend(); frame_str++) {
    if (std::regex_search(*frame_str, match, rgx)) {
      break;
    }
  }
  TORCH_INTERNAL_ASSERT(frame_str != frame_strs.rend() && !match.empty());
  return Symbol::aten(match[1].str());
}

void planMemoryWithTracing(
    std::shared_ptr<Graph>& graph,
    Strategy strat,
    std::vector<MemEvent> allocation_traces) {
  // validate
  TORCH_INTERNAL_ASSERT(!allocation_traces.empty());

  std::unordered_map<std::string, MemEvent> _allocs;
  for (auto& mem_event : allocation_traces) {
    if (mem_event.type == MemEvent::EventType::Allocate) {
      _allocs.insert({mem_event.ptr_addr, mem_event});
    } else if (mem_event.type == MemEvent::EventType::Free) {
      TORCH_INTERNAL_ASSERT(
          _allocs.count(mem_event.ptr_addr) > 0 &&
          _allocs.at(mem_event.ptr_addr).type ==
              MemEvent::EventType::Allocate &&
          _allocs.at(mem_event.ptr_addr).size == mem_event.size &&
          _allocs.at(mem_event.ptr_addr).time < mem_event.time);
      _allocs.erase(mem_event.ptr_addr);
    }
  }
  TORCH_INTERNAL_ASSERT(_allocs.empty());

  std::sort(
      allocation_traces.begin(), allocation_traces.end(), [](auto a1, auto a2) {
        return a1.time < a2.time;
      });


}

void planMemory(std::shared_ptr<Graph>& graph, Strategy strat) {
  std::unordered_map<const Value*, uint64_t> managed_value_sizes;
  std::unordered_map<const Value*, LiveRange> managed_value_ranges;
  std::vector<const Node*> out_nodes;
  std::tie(out_nodes, managed_value_sizes, managed_value_ranges) =
      getManagedStuff(graph);

  std::unordered_map<LiveRange, uint64_t, live_range_hash> managed_live_ranges;
  for (const auto& item : managed_value_sizes) {
    managed_live_ranges[managed_value_ranges[item.first]] = item.second;
  }
  std::unordered_map<LiveRange, Region, live_range_hash> allocations;

  switch (strat) {
    case Strategy::NAIVE: {
      return;
    }
    case Strategy::GREEDY_BY_SIZE: {
      allocations = greedyBySize(managed_live_ranges);
      break;
    }
    case Strategy::LINEAR_SCAN: {
      allocations = linearScanHeuristic(managed_live_ranges);
      break;
    };
    case Strategy::GREEDY_BY_BREADTH: {
      allocations = greedyByOperatorBreadth(
          managed_value_sizes, managed_value_ranges, out_nodes);
      break;
    };
    default:
      return;
  }
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

  printAllocation(allocations, managed_range_values);

  GRAPH_DEBUG("\ngraph before inserting storage node\n", *graph);

  auto storage_node = insertAllocStorageNode(graph, total_size);
  GRAPH_DEBUG("\ngraph after inserting storage node\n", *graph);

  insertAllocTensorNodes(
      graph, storage_node, allocations, managed_range_values);
  GRAPH_DEBUG("\ngraph after inserting alloc nodes\n", *graph);
}
} // namespace jit
} // namespace torch
