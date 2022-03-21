#include <torch/csrc/jit/passes/onnx/autograd_function_process.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch {
namespace jit {

void inlineAutogradFunction(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* node = *it++;
    if (node->kind() == prim::PythonOp) {
      torch::jit::SubgraphUtils::unmergeSubgraph(node);
      /*auto outerGraph = node->owningGraph();
      //outerGraph->setInsertPoint(node);
      WithInsertPoint guard(node);

      auto subgraph = node->g(torch::jit::attr::Subgraph);
      subgraph->print(std::cout, 0);

      std::unordered_map<Value*, Value*> value_map;
      auto value_map_func = [&](Value* v) { return value_map.at(v); };

      for (size_t i = 0; i < node->inputs().size(); ++i) {
        value_map[subgraph->inputs().at(i)] = node->inputs().at(i);
      }
      for (auto* sub_node : subgraph->nodes()) {
        auto* new_node = outerGraph->insertNode(outerGraph->createClone(sub_node, value_map_func));
        for (size_t i = 0; i < sub_node->outputs().size(); ++i) {
          value_map[sub_node->outputs()[i]] = new_node->outputs()[i];
        }
      }
      for (size_t i = 0; i < subgraph->outputs().size(); ++i) {
        node->outputs().at(i)->replaceAllUsesWith(value_map_func(subgraph->outputs().at(i)));
      }
      node->destroy();
      outerGraph->print(std::cout, 0);*/
    }
  }
}

// This pass is to be used for ONNX conversion only.
void ONNXAutogradFunctionProcess(std::shared_ptr<Graph>& graph) {
  inlineAutogradFunction(graph->block());
}

} // namespace jit
} // namespace torch