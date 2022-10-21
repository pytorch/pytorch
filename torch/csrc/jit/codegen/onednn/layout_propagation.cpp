#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
#include <torch/csrc/jit/codegen/onednn/layout_propagation.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

bool propagate_layout_mode = true;

void LayoutPropagation(Node* n) {
  if (!LlgaGraphHelper::isLlgaSubgraph(n))
    return;

  for (auto input : n->inputs()) {
    auto prev = input->node();
    auto offset = input->offset();
    if (LlgaGraphHelper::isLlgaSubgraph(prev)) {
      bool useOpaqueLayout = true;
      for (auto& use : input->uses()) {
        if (!LlgaGraphHelper::isLlgaSubgraph(use.user)) {
          useOpaqueLayout = false;
          break;
        }
      }
      if (useOpaqueLayout) {
        LlgaNodeWrapper(prev).setOpaqueLayout(offset);
      }
    }
  }
}

void LayoutPropagation(at::ArrayRef<Block*> blocks) {
  for (Block* block : blocks)
    for (Node* node : block->nodes())
      LayoutPropagation(node);
}

void PropagateLayout(const std::shared_ptr<Graph>& graph) {
  if (PropagateLayoutEnabled())
    LayoutPropagation(graph->block());
}

bool PropagateLayoutEnabled() {
  return propagate_layout_mode;
}

bool setPropagateLayoutMode(bool mode) {
  auto old_mode = PropagateLayoutEnabled();
  propagate_layout_mode = mode;
  return old_mode;
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
