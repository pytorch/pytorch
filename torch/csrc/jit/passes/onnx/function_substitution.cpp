#include <torch/csrc/jit/passes/onnx/function_substitution.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

namespace torch {
namespace jit {

void functionCallSubstitution(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* cur = *it++;
    switch (cur->kind()) {
      case prim::CallFunction: {
        AT_ASSERT(cur->input(0)->node()->kind() == prim::Constant);
        auto function_constant = cur->input(0)->node();
        auto fun_type =
            function_constant->output()->type()->expect<FunctionType>();

        if ((fun_type->function()->qualname().qualifiedName().find(
                 "torch.nn.functional") != std::string::npos) &&
            (fun_type->function()->qualname().qualifiedName().find(
                 "interpolate") != std::string::npos)) {
          // Remove input[0] and the node that feeds into it
          auto input_node_0 = cur->input(0)->node();
          cur->removeInput(0);
          if (!input_node_0->hasUses()) {
            input_node_0->destroy();
          }
          Node* interpolate_node = block->owningGraph()->create(
              Symbol::fromQualString("aten::__interpolate"),
              {cur->inputs()},
              cur->outputs().size());
          interpolate_node->output()->copyMetadata(cur->output());
          interpolate_node->insertAfter(cur);
          cur->replaceAllUsesWith(interpolate_node);
          cur->removeAllInputs();
          cur->destroy();
          GRAPH_UPDATE(
              "ONNX function call substitution function: '",
              fun_type->function()->name(),
              "' to aten::__interpolate");
          GRAPH_UPDATE(
              "Function in ONNX function call substitution body: ",
              *fun_type->function()->optimized_graph());
        } else {
          // Remove input[0] and the node that feeds into it
          auto input_node_0 = cur->input(0)->node();
          cur->removeInput(0);
          if (!input_node_0->hasUses()) {
            input_node_0->destroy();
          }
          functionCallSubstitution(fun_type->function()->graph()->block());
          inlineCallTo(cur, fun_type->function(), false);
        }
      } break;
      case prim::CallMethod: {
        const std::string& name = cur->s(attr::name);
        if (auto class_type = cur->input(0)->type()->cast<ClassType>()) {
          Function& function = class_type->getMethod(name);
          if (!function.isGraphFunction()) {
            continue;
          }
          functionCallSubstitution(function.graph()->block());
          inlineCallTo(cur, &function, false);
        }
      } break;
      default: {
        for (auto b : cur->blocks()) {
          functionCallSubstitution(b);
        }
      } break;
    }
  }
}

// This pass is to be used for ONNX conversion only. The ONNX converter depends
// on a number of deprecated aten operators. These operators are removed from IR
// and replaced by the compiled python function code. However, in-order to
// maintain the behavior for ONNX conversion, we replace these function calls
// with the aten symbolic which can still be used by the ONNX converter.
void ONNXFunctionCallSubstitution(Graph& graph) {
  GRAPH_DUMP("Before function call substitution calls: ", &graph);
  functionCallSubstitution(graph.block());
  GRAPH_DUMP("After function call substitution calls: ", &graph);
}

} // namespace jit
} // namespace torch
