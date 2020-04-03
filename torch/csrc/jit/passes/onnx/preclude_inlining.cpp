#include <torch/csrc/jit/passes/onnx/preclude_inlining.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

namespace torch {
namespace jit {

const auto namespaceName = std::string("___torch_mangle_4."); // 3
const auto qualFuncName =
    std::string("__torch__.torch.nn.functional."); // + namespaceName;
const auto qualFuncName2 =
    std::string("__torch__.torch.nn.functional.") + namespaceName;

void stopInliningCalls(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* cur = *it++;
    if (cur->kind() == prim::CallFunction) {
      AT_ASSERT(cur->input(0)->node()->kind() == prim::Constant);
      auto function_constant = cur->input(0)->node();
      auto fun_type =
          function_constant->output()->type()->expect<FunctionType>();

      if (fun_type->function()->qualname().qualifiedName().find(
              ".interpolate") != std::string::npos) {
        cur->removeInput(0);
        Node* interpolate_node = block->owningGraph()->create(
            Symbol::fromQualString("aten::__interpolate"),
            {cur->inputs()},
            cur->outputs().size());
        interpolate_node->output()->copyMetadata(cur->output());
        interpolate_node->insertAfter(cur);
        cur->replaceAllUsesWith(interpolate_node);
        cur->removeAllInputs();
        cur->destroy();
      } else {
        stopInliningCalls(fun_type->function()->graph()->block());
      }
      std::cout << "block \n" << block->owningGraph()->toString();
    } else {
      for (auto b : cur->blocks()) {
        stopInliningCalls(b);
      }
    }
  }
}

// This pass is to be used for ONNX conversion only. The ONNX converter depends
// on a number of deprecated aten operators. These operators are removed from IR
// and replaced by the compiled python function code. However, in-order to
// maintain the behavior for ONNX conversion, we replace these function calls
// with the aten symbolic which can still be used by the ONNX converter.
void StopInliningForONNX(Graph& graph) {
  GRAPH_DUMP("Before stop-inlining calls: ", &graph);
  stopInliningCalls(graph.block());
  GRAPH_DUMP("After stop-inlining calls: ", &graph);
}

} // namespace jit
} // namespace torch