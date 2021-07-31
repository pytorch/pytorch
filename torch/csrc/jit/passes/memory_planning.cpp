
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/memory_planning.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

c10::TensorTypePtr checkIsSupported(const Value& value) {
  TORCH_CHECK(
      value.type()->cast<TensorType>(),
      "out isn't a tensortype ",
      *value.type());
  auto ttp = value.type()->expect<TensorType>();
  TORCH_CHECK(
      ttp->scalarType().has_value(),
      "This output was profiled but didn't have a scalar type.")
  TORCH_CHECK(
      ttp->sizes().concrete_sizes().has_value(),
      "This output was profiled but doesn't have sizes.")
  return ttp;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> getSizesStrides(
    const c10::TensorTypePtr& ttp) {
  auto sizes = ttp->sizes().concrete_sizes().value();
  std::vector<int64_t> strides;
  auto _strides = ttp->strides().concrete_sizes();
  if (_strides.has_value() && _strides.value()[0] != 0) {
    strides = _strides.value();
  } else {
    strides = at::detail::defaultStrides(sizes);
  }
  return std::make_pair(sizes, strides);
}

size_t computeStorageSize(const c10::TensorTypePtr& ttp) {
  std::vector<int64_t> sizes, strides;
  std::tie(sizes, strides) = getSizesStrides(ttp);
  return at::detail::computeStorageNbytes(
      sizes, strides, elementSize(ttp->scalarType().value()));
}

std::pair<size_t, std::map<Value*, c10::TensorTypePtr>>
computeAlignedStorageSize(std::vector<Node*>& nodes) {
  // TODO: liveness analysis
  std::map<Value*, c10::TensorTypePtr> managed_tensors;
  size_t managed_bytes = 0;
  for (const auto& node : nodes) {
    for (const auto& output : node->outputs()) {
      auto ttp = checkIsSupported(*output);
      managed_bytes += computeStorageSize(ttp);
      managed_tensors.insert({output, ttp});
    }
  }

  auto aligned_size = MemoryPlanner::compute_aligned_tensor_size(managed_bytes);
  return std::make_pair(aligned_size, managed_tensors);
}

DeviceType findDeviceType(std::shared_ptr<Graph>& graph) {
  for (const auto& inp : graph->inputs()) {
    auto typ = inp->type()->cast<TensorType>();
    if (typ && typ->device().has_value()) {
      return typ->device().value().type();
    }
  }

  for (const auto& node : graph->nodes()) {
    for (const auto& inp : node->inputs()) {
      auto typ = inp->type()->cast<TensorType>();
      if (typ && typ->device().has_value()) {
        return typ->device().value().type();
      }
    }
  }
  TORCH_CHECK(false, "couldn't find device type")
}

Node* insertAllocStorageNode(std::shared_ptr<Graph>& graph, size_t total_size) {
  auto deviceType = findDeviceType(graph);
  auto* storage = graph->create(prim::AllocateStorage, 1);
  // TODO: pass device type here
  storage->i_(attr::total_size, total_size);
  storage->i_(attr::device, static_cast<int8_t>(deviceType));
  storage->insertBefore(graph->nodes().front());
  return storage;
}

void insertAllocTensorNodes(
    std::shared_ptr<Graph>& graph,
    std::vector<Node*> out_nodes,
    Node* storage,
    std::map<Value*, c10::TensorTypePtr>& managed_tensors) {
  size_t offset = 0;
  size_t total_size = storage->i(attr::total_size);
  for (auto* node : out_nodes) {
    auto* alloc = graph->create(prim::AllocateTensor, 1);
    node->addInput(alloc->output());
    GRAPH_DEBUG("inserting allocation op for ", node->getOperator().schema());
    alloc->insertBefore(node);
    alloc->addInput(storage->output());

    TORCH_CHECK(
        managed_tensors.count(node->output()) > 0,
        "output value not managed by memory planner.");
    auto ttp = managed_tensors.at(node->output());
    auto size = computeStorageSize(ttp);
    std::vector<int64_t> sizes, strides;
    std::tie(sizes, strides) = getSizesStrides(ttp);
    TORCH_CHECK(offset + size <= total_size, "not enough memory");
    alloc->i_(attr::size, size);
    alloc->i_(attr::offset, offset);
    alloc->is_(attr::sizes, sizes);
    alloc->is_(attr::stride, strides);
    alloc->i_(attr::device, static_cast<int8_t>(storage->i(attr::device)));
    alloc->i_(attr::dtype, static_cast<int8_t>(ttp->scalarType().value()));

    offset += size;
  }
  TORCH_CHECK(offset == total_size, "exceeded total slab size");
  GRAPH_DEBUG(offset, " bytes of entire slab ", total_size, " filled");
}

std::vector<Node*> findOutVariantNodes(std::shared_ptr<Graph>& graph) {
  std::vector<Node*> out_nodes = {};
  for (auto* node : graph->nodes()) {
    for (const auto& variant : getAllOperatorsFor(node->kind())) {
      auto variant_args = variant->schema().arguments();
      auto maybe_out_arg =
          std::find_if(variant_args.begin(), variant_args.end(), [](auto arg) {
            return arg.name() == "out";
          });
      if (maybe_out_arg != variant_args.end()) {
        out_nodes.emplace_back(node);
      }
    }
  }
  return out_nodes;
}

void planMemory(std::shared_ptr<Graph>& graph) {
  auto out_nodes = findOutVariantNodes(graph);
  size_t aligned_size;
  std::map<Value*, c10::TensorTypePtr> managed_tensors;
  std::tie(aligned_size, managed_tensors) =
      computeAlignedStorageSize(out_nodes);
  GRAPH_DEBUG("\ngraph before inserting storage node\n", *graph);

  auto storage_node = insertAllocStorageNode(graph, aligned_size);
  GRAPH_DEBUG("\ngraph after inserting storage node\n", *graph);

  insertAllocTensorNodes(graph, out_nodes, storage_node, managed_tensors);
  GRAPH_DEBUG("\ngraph after inserting alloc nodes\n", *graph);
}

} // namespace jit
} // namespace torch