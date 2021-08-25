#include <torch/csrc/jit/passes/variadic_ops.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/remove_mutation.h>

namespace torch {
namespace jit {

namespace {

class VariadicUpdater {
 public:
  explicit VariadicUpdater(
      std::shared_ptr<Graph> graph,
      NodeKind op,
      NodeKind variadic_op)
      : graph_(std::move(graph)), op_(op), variadic_op_(variadic_op) {}

  bool run() {
    collectOpNodes(graph_->block());
    bool changed = false;
    for (auto n : op_nodes_) {
      changed |= replaceWithVariadicOp(n);
    }
    return changed;
  }

 private:
  void collectOpNodes(Block* block) {
    for (auto node : block->nodes()) {
      if (node->kind() == op_) {
        op_nodes_.push_back(node);
      }
      for (Block* b : node->blocks()) {
        collectOpNodes(b);
      }
    }
  }

  bool replaceWithVariadicOp(Node* op_node) {
    if (op_node->input(0)->node()->kind() != prim::ListConstruct) {
      return false;
    }
    auto list = op_node->input(0)->node();
    // We do not transform ops whose list input can not be moved to the
    // position before op. This in turn implies that there is some mutation
    // of the input list before op.
    if (!getOrCreateAliasDb()->couldMoveBeforeTopologically(list, op_node)) {
      return false;
    }
    std::vector<Value*> inputs = list->inputs().vec();
    // Add non-list inputs
    for (size_t i = 1; i < op_node->inputs().size(); ++i) {
      inputs.push_back(op_node->input(i));
    }
    auto var_op_node = op_node->owningGraph()->create(variadic_op_, inputs);
    GRAPH_UPDATE("Adding\n", *var_op_node);
    var_op_node->insertBefore(op_node);
    GRAPH_UPDATE("Replacing\n", *op_node, "with\n", *var_op_node);
    op_node->output()->replaceAllUsesWith(var_op_node->output());
    GRAPH_UPDATE("Deleting\n", *op_node);
    op_node->destroy();
    if (!list->hasUses()) {
      GRAPH_UPDATE("Deleting\n", *list);
      list->destroy();
    }
    return true;
  }

  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;

  std::vector<Node*> op_nodes_;

  NodeKind op_;
  NodeKind variadic_op_;
};

} // namespace

bool UseVariadicOp(
    const std::shared_ptr<Graph>& graph,
    NodeKind op,
    NodeKind variadic_op) {
  const std::string pass_name = std::string("variadic ") + op.toQualString();
  GRAPH_DUMP("Before " + pass_name, graph);
  bool changed = VariadicUpdater(graph, op, variadic_op).run();
  if (changed) {
    GRAPH_DUMP("After " + pass_name, graph);
  }
  return changed;
}

bool RemoveListMutationAndUseVariadicOp(
    const std::shared_ptr<Graph>& graph,
    NodeKind op,
    NodeKind variadic_op) {
  bool changed_in_last_iter = true;
  bool changed = false;
  while (changed_in_last_iter) {
    changed_in_last_iter = RemoveListMutation(graph);
    changed_in_last_iter =
        UseVariadicOp(graph, op, variadic_op) || changed_in_last_iter;
    changed = changed || changed_in_last_iter;
  }
  return changed;
}

bool UseVariadicCat(const std::shared_ptr<Graph>& graph) {
  return UseVariadicOp(graph, aten::cat, prim::VarConcat);
}

bool RemoveListMutationAndUseVariadicCat(const std::shared_ptr<Graph>& graph) {
  return RemoveListMutationAndUseVariadicOp(graph, aten::cat, prim::VarConcat);
}

bool UseVariadicStack(const std::shared_ptr<Graph>& graph) {
  return UseVariadicOp(graph, aten::stack, prim::VarStack);
}

bool RemoveListMutationAndUseVariadicStack(
    const std::shared_ptr<Graph>& graph) {
  return RemoveListMutationAndUseVariadicOp(graph, aten::stack, prim::VarStack);
}

} // namespace jit
} // namespace torch
