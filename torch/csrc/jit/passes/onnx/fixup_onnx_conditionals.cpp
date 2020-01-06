#include <torch/csrc/jit/passes/onnx/fixup_onnx_conditionals.h>

namespace torch {
namespace jit {

namespace onnx{
using namespace ::c10::onnx;
}

bool FixupONNXIfs(Block* block) {
  bool has_nodes = false;
  for (auto* node : block->nodes()) {
    has_nodes = true;
    if (node->kind() == ::c10::onnx::If) {
      auto* if_node = node;
      auto* graph = if_node->owningGraph();
      for (Block* block : node->blocks()) {
        if (!FixupONNXIfs(block)) {
          //ONNX does not support empty blocks, must use some op which does nothing
          Value* output = block->outputs()[0];
          Node* id_node = graph->create(onnx::Identity);
          id_node->insertBefore(block->return_node());
          id_node->addInput(output);
          block->return_node()->replaceInputWith(output, id_node->output());
        }
      }
    }
    else {
      for (Block* block : node->blocks()) {
        FixupONNXIfs(block);
      }
    }
  }
  return has_nodes;
}

void FixupONNXConditionals(std::shared_ptr<Graph>& graph) {
  FixupONNXIfs(graph->block());
}

} //jit
} //torch