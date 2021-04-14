#include <torch/csrc/jit/backends/generate_debug_handles.h>

#include <stack>

namespace torch {
namespace jit {

namespace {
bool isGraphInlined(const Graph& graph) {
  std::stack<const Block*> blocks_to_visit;
  blocks_to_visit.push(graph.block());
  while (!blocks_to_visit.empty()) {
    const Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (const Node* n : b->nodes()) {
      if (n->kind() == prim::CallMethod || n->kind() == prim::CallFunction) {
        // In order to generate debug handles we expect inlined function.
        // Reason being is that we will be returning a map of debug handles to
        // inlined callstack pointers.
        // Here we check if we have any callMethod and callFunction nodes
        // which are graph functions. If so graph has not been inlined.
        // Graph function is the one where Function of Module's method
        // has graph corresponding to it.
        // This is opposed to pure Functions that directly bind to
        auto function_constant = n->input(0)->node();
        auto fun_type =
            function_constant->output()->type()->expect<FunctionType>();
        if (fun_type->function()->isGraphFunction()) {
          return false;
        }
      }
      for (const Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  return true;
}
} // namespace

NodeToDebugHandle TORCH_API generate_debug_handles(const std::shared_ptr<Graph>& graph) {
  NodeToDebugHandle node_to_debug_handles;
  BackendDebugHandleManager* dbg_handle_manager_ptr = getBackendDebugHandleManager();

  TORCH_CHECK(isGraphInlined(*graph),
      "Debug handles and InlinedCallStackPtrMap can be generated "
      "only for inlined graphs.");
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      DebugHandleType debug_handle{-1};
      if (dbg_handle_manager_ptr) {
        if (n->callstack().has_value()) {
          debug_handle =
            dbg_handle_manager_ptr->getNextDebugHandleForInlinedCallStackPtr(
                n->sourceRange(), n->callstack().value());
        } else {
          // If node has no callstack, it is the top level node.
          // In that case just save source range.
          debug_handle = dbg_handle_manager_ptr
                             ->getNextDebugHandleForInlinedCallStackPtr(
                                 n->sourceRange(),
                                 c10::intrusive_ptr<InlinedCallStack>());
        }
      }
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
