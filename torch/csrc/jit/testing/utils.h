#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace testing {

std::vector<Node*> findAllNodes(
    c10::ArrayRef<torch::jit::Block*> blocks,
    Symbol kind,
    bool recurse = true) {
  std::vector<Node*> ret;
  for (Block* block : blocks) {
    for (Node* n : block->nodes()) {
      if (n->kind() == kind) {
        ret.push_back(n);
      }
      if (recurse) {
        auto nodes = findAllNodes(n->blocks(), kind, recurse);
        ret.insert(ret.end(), nodes.begin(), nodes.end());
      }
    }
  }
  return ret;
}

std::vector<Node*> findAllNodes(
    Block* block,
    Symbol kind,
    bool recurse = true) {
  std::vector<Block*> blocks = {block};
  return findAllNodes(blocks, kind, recurse);
}

Node* findNode(
    c10::ArrayRef<torch::jit::Block*> blocks,
    Symbol kind,
    bool recurse = true) {
  for (Block* block : blocks) {
    for (Node* n : block->nodes()) {
      if (n->kind() == kind) {
        return n;
      }
      if (recurse) {
        auto node = findNode(n->blocks(), kind, recurse);
        if (node != nullptr) {
          return node;
        }
      }
    }
  }
  return nullptr;
}

Node* findNode(Block* block, Symbol kind, bool recurse = true) {
  std::vector<Block*> blocks = {block};
  return findNode(blocks, kind, recurse);
}

Node* findNode(Graph& g, Symbol kind, bool recurse = true) {
  return findNode(g.block(), kind, recurse);
}

std::vector<Node*> findAllNodes(Graph& g, Symbol kind, bool recurse = true) {
  return findAllNodes(g.block(), kind, recurse);
}

} // namespace testing
} // namespace jit
} // namespace torch
