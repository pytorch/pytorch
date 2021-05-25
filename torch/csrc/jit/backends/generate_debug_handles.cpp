#include <torch/csrc/jit/backends/generate_debug_handles.h>

#include <stack>

namespace torch {
namespace jit {

NodeToDebugHandle generate_debug_handles(const std::shared_ptr<Graph>& graph) {
  NodeToDebugHandle node_to_debug_handles;
  auto* debug_info_recorder_ptr = getBackendDebugInfoRecorder();

  // Note now we make having a valid debug_handle_manager a must.
  // This avoids silently failing when say some code change results in
  // to_backend not creating appropriate debug_handle_manager to
  // be used with backend's preprocess function.
  TORCH_CHECK(
      debug_info_recorder_ptr, "Valid debug info recorder must be available.");
  std::stack<Block*> blocks_to_visit;
  // TODO: Look into using DepthFirstGraphNodeIterator
  // At the moment it takes non-const graph but maybe we can make it
  // general such that it can work with both.
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      DebugHandleType debug_handle =
          debug_info_recorder_ptr->getNextDebugHandle(n);
      node_to_debug_handles.emplace(n, debug_handle);
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  return node_to_debug_handles;
}

} // namespace jit
} // namespace torch
