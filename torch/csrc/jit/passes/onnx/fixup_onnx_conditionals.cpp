#include <torch/csrc/jit/passes/onnx/fixup_onnx_conditionals.h>

namespace torch {
namespace jit {

namespace onnx{
using namespace ::c10::onnx;
}

void FixupONNXIfs(Block* block) {
  for (auto* node : block->nodes()) {
    if (node->kind() == ::c10::onnx::If) {
      auto* if_node = node;
      auto* graph = if_node->owningGraph();
      for (Block* block : node->blocks()) {
        FixupONNXIfs(block);
        if (block->nodes().begin() == block->nodes().end()) {
          //ONNX does not support empty blocks, must use some op which does nothing
          Value* output = block->outputs()[0];
          Node* id_node = graph->create(onnx::Identity);
          id_node->insertBefore(block->return_node());
          id_node->addInput(output);
          id_node->output()->copyMetadata(output);
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
}

void FixupONNXConditionals(std::shared_ptr<Graph>& graph) {
  FixupONNXIfs(graph->block());
}

} //jit
} //torch
