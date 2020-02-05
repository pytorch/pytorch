#include <torch/csrc/jit/passes/onnx/fixup_jit_exceptions.h>

namespace torch {
namespace jit {

namespace onnx{
using namespace ::c10::onnx;
}

bool IsExceptionBlock(Block* block) {
  for (auto* output : block->outputs()) {
    if (output->node()->kind() == ::c10::prim::Uninitialized) {
      return true;
    }
  }
  return false;
}
void RemoveUninitialized(Block* block) {
  std::vector<Node*> to_destroy;
  for (auto* node : block->nodes()) {
    if (node->kind() == ::c10::prim::Uninitialized) {
      to_destroy.push_back(node);
    }
    for (Block* sub_block : node->blocks()) {
      RemoveUninitialized(sub_block);
    }
  }
  for (auto* node : to_destroy) {
    node->destroy();
  }
}

bool RemoveExceptionBranches(Block* block) {
  bool pathRemoved = false;
  std::vector<Node*> to_destroy;
  for (auto* node : block->nodes()) {
    for (Block* sub_block : node->blocks()) {
      pathRemoved |= RemoveExceptionBranches(sub_block);
    }
    if (node->kind() == ::c10::prim::If) {
      auto* graph = node->owningGraph();
      Block* valid_path = nullptr;
      bool foundException = false;
      for (Block* sub_block : node->blocks()) {
        if (IsExceptionBlock(sub_block)) {
          foundException = true;
        }
        else {
          valid_path = sub_block;
        }
      }
      if (foundException && valid_path) {
        std::vector<Node*> tmp;
        for (auto* valid_node : valid_path->nodes()) {
          tmp.push_back(valid_node);
        }
        Node* cur = node;
        for (auto* valid_node : tmp) {
          valid_node->moveAfter(cur);
          cur = valid_node;
        }
        while(!node->blocks().empty()) {
          node->eraseBlock(0);
        }
        node->replaceAllUsesWith(cur);
        to_destroy.push_back(node);
      }
    }
  }
  for (auto* node : to_destroy) {
    TORCH_WARN("Eliminating branch which throws an exception, removing Node: ", *node);
    pathRemoved = true;
    node->destroy();
  }
  return pathRemoved;
}

void HandleExceptionBranches(Block* block) {
  if (RemoveExceptionBranches(block)) {
    RemoveUninitialized(block);
  }
}

void FixupJitExceptions(std::shared_ptr<Graph>& graph) {
  HandleExceptionBranches(graph->block());
}

} //jit
} //torch
