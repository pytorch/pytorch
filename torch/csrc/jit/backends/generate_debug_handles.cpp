#include <torch/csrc/jit/backends/generate_debug_handles.h>

#include <stack>

namespace torch {
namespace jit {

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

std::pair<NodeToDebugHandle, DebugHandleToDebugInfo> generate_debug_handles(
    const Module& mod, const std::vector<std::string>& method_names) {
  NodeToDebugHandle node_to_debug_handles;
  BackendDebugHandleManager dbg_handle_manager;

  for (const auto& method_name : method_names) {
    auto m = mod.find_method(method_name);
    if (m) {
      const auto& graph = m.value().graph();
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
          if (n->callstack().has_value()) {
            debug_handle =
              dbg_handle_manager.getNextDebugHandleForInlinedCallStackPtr(
                  n->sourceRange(), n->callstack().value());
          } else {
            // If node has no callstack, it is the top level node.
            // In that case just save source range.
            debug_handle = dbg_handle_manager
                               .getNextDebugHandleForInlinedCallStackPtr(
                                   n->sourceRange(),
                                   c10::intrusive_ptr<InlinedCallStack>());
          }
          node_to_debug_handles.emplace(debug_handle, n);
          for (Block* subblock : n->blocks()) {
            blocks_to_visit.push(subblock);
          }
        }
      }
    }
  }
  return {node_to_debug_handles, dbg_handle_manager.getCallStackPtrMap()};
}

} // namespace jit
} // namespace torch
