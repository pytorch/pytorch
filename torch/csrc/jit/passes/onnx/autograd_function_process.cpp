#include <torch/csrc/jit/passes/onnx/autograd_function_process.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

namespace torch {
namespace jit {

void inlineAutogradFunction(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* node = *it++;
    if (node->kind() == prim::PythonOp) {
      auto subgraph = node->g(torch::jit::attr::Subgraph);
      subgraph->print(std::cout, 0);
    }
    /*switch (cur->kind()) {
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
          interpolate_node->copyMetadata(cur);
          cur->replaceAllUsesWith(interpolate_node);
          cur->removeAllInputs();
          cur->destroy();
          GRAPH_UPDATE(
              "ONNX function call substitution function: '",
              fun_type->function()->name(),
              "' to aten::__interpolate");
          GRAPH_UPDATE(
              "Function in ONNX function call substitution body: ",
              toGraphFunction(*fun_type->function()).optimized_graph());
        } else {
          // Remove input[0] and the node that feeds into it
          auto input_node_0 = cur->input(0)->node();
          cur->removeInput(0);
          if (!input_node_0->hasUses()) {
            input_node_0->destroy();
          }
          auto& graphFunction = toGraphFunction(*fun_type->function());
          functionCallSubstitution(graphFunction.graph()->block());
          inlineCallTo(cur, &graphFunction, false);
        }
      } break;
      case prim::CallMethod: {
        const std::string& name = cur->s(attr::name);
        if (auto class_type = cur->input(0)->type()->cast<ClassType>()) {
          Function& function = class_type->getMethod(name);
          if (auto graphFunction = tryToGraphFunction(function)) {
            functionCallSubstitution(graphFunction->graph()->block());
            inlineCallTo(cur, graphFunction, false);
          }
        }
      } break;
      default: {
        for (auto b : cur->blocks()) {
          functionCallSubstitution(b);
        }
      } break;
    }*/
  }
}

// This pass is to be used for ONNX conversion only.
void ONNXAutogradFunctionProcess(std::shared_ptr<Graph>& graph) {
  inlineAutogradFunction(graph->block());
}

} // namespace jit
} // namespace torch