#include <torch/csrc/jit/codegen/onednn/defer_size_check.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry_util.h>

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
        uses.size() > 1 && std::all_of(uses.begin(), uses.end(), [&](auto& u) {
          if (u.user == node) {
            return true;
          }
          // match with shape-preserving unary ops in
          // tensorexpr_elementwise_set that's defined in
          // torch/csrc/jit/runtime/symbolic_shape_registry_util.cpp
          OperatorMap<std::string> schemaMap = get_tensorexpr_elementwise_set();
          std::optional<std::string> mapping =
              schemaMap.find(u.user->getOperator());
          return mapping == "unary";
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
