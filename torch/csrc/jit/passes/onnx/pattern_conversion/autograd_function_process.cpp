#include <torch/csrc/jit/passes/onnx/pattern_conversion/autograd_function_process.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch {
namespace jit {

void convertSubgraphToSubBlock(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* node = *it++;
    if (node->kind() == prim::PythonOp) {
      // Construct subblock
      auto subblock = node->addBlock();
      auto graph = subblock->owningGraph();

      std::unordered_map<Value*, Value*> env;
      // Populate subblock with subgraph nodes
      auto subgraph = node->g(attr::Subgraph);
      for (const auto i : c10::irange(subgraph->inputs().size())) {
        subblock->addInput()->copyMetadata(subgraph->inputs()[i]);
        env[subgraph->inputs()[i]] = subblock->inputs()[i];
      }
      for (auto* n : subgraph->nodes()) {
        auto cloned_n =
            subblock->appendNode(graph->createClone(n, [&](Value* v) {
              return env.find(v) != env.end() ? env[v] : v;
            }));
        for (size_t i = 0; i < n->outputs().size(); ++i) {
          env[n->outputs().at(i)] = cloned_n->outputs().at(i);
          auto it = std::find(
              subgraph->outputs().begin(),
              subgraph->outputs().end(),
              n->outputs()[i]);
          if (it != subgraph->outputs().end()) {
            subblock->registerOutput(cloned_n->outputs()[i]);
          }
        }
      }
      // Remove subgraph attribute from the pythonOp node and recurse through
      // sub-blocks
      node->removeAttribute(attr::Subgraph);
    }
    for (auto block : node->blocks()) {
      convertSubgraphToSubBlock(block);
    }
  }
}

// This pass is to be used for ONNX conversion only.
void ONNXAutogradFunctionProcess(std::shared_ptr<Graph>& graph) {
  convertSubgraphToSubBlock(graph->block());
}

} // namespace jit
} // namespace torch
