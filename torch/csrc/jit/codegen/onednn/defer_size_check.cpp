#include <torch/csrc/jit/ir/alias_analysis.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

class SizeCheckMover {
 private:
  Block* block_;
  std::shared_ptr<Graph> graph_;

 public:
  SizeCheckMover(Block* block, std::shared_ptr<Graph> graph)
      : block_(block), graph_(std::move(graph)) {}

  bool analyzeNode(Node* node, AliasDb& aliasDb) {
    //
    // %b = addmm(%a)
    // %sz = aten::size(%b)
    // %c = relu(%b)
    //  =>
    // %b = addmm(%a)
    // %c = relu(%b)
    // %sz = aten::size(%c)
    //       ^-- move size check after relu as it preserves input shape
    //
    if (!node->matches("aten::size(Tensor self) -> int[]"))
      return false;

    auto* input = node->input(0);
    auto& uses = input->uses();
    bool onlyUsedByShapePreserveOp =
        uses.size() > 1 &&
        std::all_of(uses.begin(), uses.end(), [node](auto& u) {
          return u.user == node ||
              // TODO: register more shape preserved op
              u.user->matches("aten::relu(Tensor self) -> Tensor") ||
              u.user->matches("aten::sigmoid(Tensor self) -> Tensor");
        });

    if (!onlyUsedByShapePreserveOp)
      return false;

    for (const auto& use : uses) {
      if (use.user == node)
        continue;
      auto shapePreserveOp = use.user;
      if (aliasDb.moveAfterTopologicallyValid(node, shapePreserveOp)) {
        node->replaceInputWith(input, shapePreserveOp->output(0));
        return true;
      }
    }

    return false;
  }

  void run() {
    bool changed = true;
    while (changed) {
      changed = false;
      AliasDb aliasDb(graph_);
      for (Node* node : block_->nodes()) {
        changed |= analyzeNode(node, aliasDb);
      }
    }

    for (Node* node : block_->nodes())
      for (Block* subBlock : node->blocks())
        SizeCheckMover(subBlock, graph_).run();
  }
};

void DeferSizeCheck(std::shared_ptr<Graph>& graph) {
  SizeCheckMover(graph->block(), graph).run();
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
