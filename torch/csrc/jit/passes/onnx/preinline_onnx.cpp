#include <torch/csrc/jit/passes/onnx/preinline_onnx.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

namespace torch {
namespace jit {

void replaceFunctions(Node* to_replace, Function* callee) {
  if (callee->name() == "interpolate") {
    to_replace->removeInput(0);
    Node* interpolate_node = to_replace->owningGraph()->create(
        Symbol::fromQualString("aten::__interpolate"),
        {to_replace->inputs()},
        to_replace->outputs().size());
    interpolate_node->output()->copyMetadata(to_replace->output());
    interpolate_node->insertAfter(to_replace);
    to_replace->replaceAllUsesWith(interpolate_node);
    to_replace->removeAllInputs();
    to_replace->destroy();
    return;
  }
}

void PreInlineCalls(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* cur = *it++;
    switch (cur->kind()) {
      case prim::CallFunction: {
        AT_ASSERT(cur->input(0)->node()->kind() == prim::Constant);
        auto function_constant = cur->input(0)->node();
        auto fun_type =
            function_constant->output()->type()->expect<FunctionType>();
        replaceFunctions(cur, fun_type->function());
      } break;
      default: {
        for (auto b : cur->blocks()) {
          PreInlineCalls(b);
        }
      } break;
    }
  }
}

void PreInlineONNX(Graph& graph) {
  GRAPH_DUMP("Before Pre-inlining: ", &graph);
  PreInlineCalls(graph.block());
  GRAPH_DUMP("After Pre-inlining: ", &graph);
}

} // namespace jit
} // namespace torch