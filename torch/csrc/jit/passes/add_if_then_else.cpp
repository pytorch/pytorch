#include <torch/csrc/jit/passes/add_if_then_else.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

namespace torch {
namespace jit {

namespace {

bool hasNoNodes(Block* block) {
  auto nodes = block->nodes();
  return nodes.begin() == nodes.end();
}

bool hasTrivialSubBlocks(Node* node) {
  const auto blocks = node->blocks();
  TORCH_DCHECK_EQ(blocks.size(), 2);

  return hasNoNodes(blocks[0]) && hasNoNodes(blocks[1]);
}

} // namespace

bool AddIfThenElseOp(std::shared_ptr<Graph>& graph) {
  std::vector<Node*> to_replace;
  DepthFirstGraphNodeIterator graph_it(graph);
  for (auto* node = graph_it.next(); node != nullptr; node = graph_it.next()) {
    if (node->kind() != prim::If) {
      continue;
    }
    if (node->outputs().size() != 1) {
      continue;
    }
    if (hasTrivialSubBlocks(node)) {
      to_replace.push_back(node);
    }
  }

  for (auto* node : to_replace) {
    auto* if_then_else_node = graph->create(prim::IfThenElse, 1);
    if_then_else_node->addInput(node->input());
    auto blocks = node->blocks();
    if_then_else_node->addInput(blocks[0]->return_node()->input());
    if_then_else_node->addInput(blocks[1]->return_node()->input());

    if_then_else_node->insertBefore(node);
    if_then_else_node->output()->copyMetadata(node->output());

    node->output()->replaceAllUsesWith(if_then_else_node->output());
    node->destroy();
  }
  return !to_replace.empty();
}

} // namespace jit
} // namespace torch
