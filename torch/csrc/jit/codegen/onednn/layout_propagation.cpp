#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
#include <torch/csrc/jit/codegen/onednn/layout_propagation.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

void LayoutPropagation(Node* n) {
  if (!LlgaGraphHelper::isLlgaSubgraph(n))
    return;

  // initial attr::output_layouts if undefined
  if (!n->hasAttribute(attr::output_layouts)) {
    const auto num_output = n->outputs().size();
    GRAPH_DEBUG("Initial output_layouts of size ", num_output);
    std::vector<int64_t> layouts(num_output, 0);
    n->is_(attr::output_layouts, layouts);
  }

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
  LayoutPropagation(graph->block());
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
