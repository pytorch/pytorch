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
      NodeKind variadic_op,
      size_t list_idx = 0)
      : graph_(std::move(graph)),
        op_(op),
        variadic_op_(variadic_op),
        list_idx_(list_idx) {}

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
    const size_t num_inputs = op_node->inputs().size();
    TORCH_CHECK(list_idx_ < num_inputs);
    if (op_node->input(list_idx_)->node()->kind() != prim::ListConstruct) {
      return false;
    }
    auto list = op_node->input(list_idx_)->node();
    const size_t list_len = list->inputs().size();

    // We do not transform ops whose list input can not be moved to the
    // position before op. This in turn implies that there is some mutation
    // of the input list before op.
    if (!getOrCreateAliasDb()->couldMoveBeforeTopologically(list, op_node)) {
      return false;
    }

    // Construct new inputs
    std::vector<Value*> inputs;
    inputs.reserve(num_inputs + list_len - 1);
    inputs.insert(
        inputs.end(),
        op_node->inputs().begin(),
        op_node->inputs().begin() + list_idx_);
    inputs.insert(inputs.end(), list->inputs().begin(), list->inputs().end());
    inputs.insert(
        inputs.end(),
        op_node->inputs().begin() + list_idx_ + 1,
        op_node->inputs().end());

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

  size_t list_idx_;
};

} // namespace

bool UseVariadicOp(
    const std::shared_ptr<Graph>& graph,
    NodeKind op,
    NodeKind variadic_op,
    size_t list_idx) {
  const std::string pass_name = std::string("variadic ") + op.toQualString();
  GRAPH_DUMP("Before " + pass_name, graph);
  bool changed = VariadicUpdater(graph, op, variadic_op, list_idx).run();
  if (changed) {
    GRAPH_DUMP("After " + pass_name, graph);
  }
  return changed;
}

bool RemoveListMutationAndUseVariadicOp(
    const std::shared_ptr<Graph>& graph,
    NodeKind op,
    NodeKind variadic_op,
    size_t list_idx) {
  bool changed_in_last_iter = true;
  bool changed = false;
  while (changed_in_last_iter) {
    changed_in_last_iter = RemoveListMutation(graph);
    changed_in_last_iter =
        UseVariadicOp(graph, op, variadic_op, list_idx) || changed_in_last_iter;
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
