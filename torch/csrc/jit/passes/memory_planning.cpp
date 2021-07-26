
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/memory_planning.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

size_t computeStorageSize(std::unique_ptr<ProfilingRecord>& pr) {
  //  std::unordered_set<const Value*> managed_tensor_values;
  size_t managed_bytes = 0;
  for (const auto& frame_id_bytes_per_ts : pr->nbytes_per_frame_) {
    auto _frame_id = frame_id_bytes_per_ts.first;
    auto bytes_per_ts = frame_id_bytes_per_ts.second;
    for (const auto& outputs_nbytess : bytes_per_ts) {
      //      auto out_v = outputs_nbytess.first;
      auto nbytes = outputs_nbytess.second;
      //      managed_tensor_values.emplace(out_v);
      managed_bytes += nbytes;
    }
  }

  auto aligned_size = MemoryPlanner::compute_aligned_tensor_size(managed_bytes);
  return aligned_size;
}

// (needsProfiledInputs(n) || needsProfiledOutput(o->node())))
Node* insertAllocStorageNode(std::shared_ptr<Graph>& graph, size_t size) {
  // insert slab allocation node
  auto* storage = graph->create(prim::AllocateStorage, 1);
  // TODO: pass device type here
  storage->i_(attr::size, size); //->ty_(attr::device, at::kCPU);
  storage->insertBefore(graph->nodes().front());
  return storage;
}

void insertAllocTensorNodes(std::shared_ptr<Graph>& graph, Node* storage) {
  for (auto* node : graph->nodes()) {
    for (const auto& variant : getAllOperatorsFor(node->kind())) {
      auto variant_args = variant->schema().arguments();
      auto maybe_out_arg = std::find_if(
          variant_args.begin(), variant_args.end(), [](auto arg) {
            return arg.name() == "out";
          });
      if (maybe_out_arg != variant_args.end()) {
        GRAPH_DEBUG("inserting allocation op before ", node->kind());
        // insert prim::allocate nodes
        // TODO: put here all of the attrs *for* the output tensor
        auto* alloc = graph->create(prim::AllocateTensor, 1);
        alloc->insertBefore(node);
        // pass slab to allocation nodes
        alloc->addInput(storage->output());
        // pass allocated region to out variant of op
        node->addInput(alloc->output());
      }
    }
  }
}

void InsertAllocationNodes(
    std::shared_ptr<Graph>& graph,
    std::unique_ptr<ProfilingRecord>& pr) {
  // count max bytes mem
  auto aligned_size = computeStorageSize(pr);
  // insert allocation nodes
  auto storage_node = insertAllocStorageNode(graph, aligned_size);
  insertAllocTensorNodes(graph, storage_node);
}

} // namespace jit
} // namespace torch