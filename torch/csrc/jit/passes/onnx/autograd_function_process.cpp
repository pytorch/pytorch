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
    for (auto block : node->blocks()) {
      inlineAutogradFunction(block);
    }
    if (node->kind() == prim::PythonOp) {
      torch::jit::SubgraphUtils::unmergeSubgraph(node);
    }
  }
}

// This pass is to be used for ONNX conversion only.
void ONNXAutogradFunctionProcess(std::shared_ptr<Graph>& graph) {
  inlineAutogradFunction(graph->block());
}

} // namespace jit
} // namespace torch