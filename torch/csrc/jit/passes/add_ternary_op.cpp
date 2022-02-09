#include <torch/csrc/jit/passes/add_ternary_op.h>
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
  DCHECK_EQ(blocks.size(), 2);

  return hasNoNodes(blocks[0]) && hasNoNodes(blocks[1]);
}

} // namespace

bool AddTernaryOp(std::shared_ptr<Graph>& graph) {
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
    auto* ternary = graph->create(prim::Ternary, 1);
    ternary->addInput(node->input());
    auto blocks = node->blocks();
    ternary->addInput(blocks[0]->return_node()->input());
    ternary->addInput(blocks[1]->return_node()->input());

    ternary->insertBefore(node);
    ternary->output()->copyMetadata(node->output());

    node->output()->replaceAllUsesWith(ternary->output());
    node->destroy();
  }
  return !to_replace.empty();
}

} // namespace jit
} // namespace torch
