#include "torch/csrc/jit/passes/onnx/fixup_onnx_loop.h"

namespace torch { namespace jit {

void FixupONNXLoops(Block *block) {
  for (auto *node : block->nodes()) {
    if (node->kind() == torch::jit::onnx::Loop) {
      JIT_ASSERT(node->blocks().size() == 1);
      auto *sub_block = node->blocks()[0];
      sub_block->insertInput(1, "cond");
    }
    for (Block * block : node->blocks()) {
      FixupONNXLoops(block);
    }
  }
}

void FixupONNXLoops(std::shared_ptr<Graph>& graph) {
  FixupONNXLoops(graph->block());
}

}}  // namespace torch::jit
