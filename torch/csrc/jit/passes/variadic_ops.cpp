#include <torch/csrc/jit/passes/variadic_ops.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/remove_mutation.h>

namespace torch::jit {

namespace {

std::vector<size_t> identifyListArgIndices(const c10::FunctionSchema& schema) {
  std::vector<size_t> list_indices;
  const auto& args = schema.arguments();
  for (const auto i : c10::irange(args.size())) {
    auto list_type = args[i].type()->castRaw<ListType>();
    if (list_type && list_type->getElementType()->castRaw<TensorType>()) {
      list_indices.push_back(i);
    }
  }
  return list_indices;
}

bool isTensorListConstruct(Node* node) {
  if (node->kind() != prim::ListConstruct) {
    return false;
  }
  const auto type = node->output()->type()->castRaw<ListType>();
  TORCH_CHECK(type != nullptr);
  const auto& elem_type = type->getElementType();
  return elem_type->castRaw<TensorType>();
}

class VariadicUpdater {
 public:
  VariadicUpdater(
      std::shared_ptr<Graph> graph,
      NodeKind op,
      NodeKind variadic_op)
      : graph_(std::move(graph)),
        alias_db_(graph_),
        op_(op),
        variadic_op_(variadic_op) {}

  bool run() {
    collectOpNodes(graph_->block());
    bool changed = false;
    for (auto n : op_nodes_) {
      changed |= replaceWithVariadicOp(n);
    }
    return changed;
  }

 private:
  void recordSchema(Node* op_node) {
    const auto& schema = op_node->schema();
    auto it = schema_to_list_indices_.find(schema.name());
    if (it == schema_to_list_indices_.end()) {
      schema_to_list_indices_.emplace(
          schema.overload_name(), identifyListArgIndices(schema));
    }
  }

  const std::vector<size_t>& getListIndices(Node* op_node) const {
    const auto& schema = op_node->schema();
    auto it = schema_to_list_indices_.find(schema.overload_name());
    TORCH_CHECK(it != schema_to_list_indices_.end());
    return it->second;
  }

  void collectOpNodes(Block* block) {
    for (auto node : block->nodes()) {
      if (node->kind() == op_) {
        op_nodes_.push_back(node);
        recordSchema(node);
      }
      for (Block* b : node->blocks()) {
        collectOpNodes(b);
      }
    }
  }

  bool allListInputsAreValid(Node* op_node) {
    const size_t num_inputs = op_node->inputs().size();
    for (const auto list_idx : getListIndices(op_node)) {
      TORCH_CHECK(list_idx < num_inputs);
      const auto list = op_node->input(list_idx)->node();
      // We do not transform ops whose list input can not be moved to the
      // position before op. This in turn implies that there is some mutation
      // of the input list before op.
      if (!isTensorListConstruct(list) ||
          !alias_db_.couldMoveBeforeTopologically(list, op_node)) {
        return false;
      }
    }
    return true;
  }

  void insertAllInputsBetween(
      std::vector<Value*>& inputs,
      Node* node,
      size_t start_idx,
      size_t end_idx) const {
    const size_t num_inputs = node->inputs().size();
    TORCH_CHECK(start_idx <= end_idx && end_idx <= num_inputs);
    inputs.insert(
        inputs.end(),
        node->inputs().begin() + start_idx,
        node->inputs().begin() + end_idx);
  }

  void insertIntegerInput(std::vector<Value*>& inputs, size_t input) {
    auto constant = graph_->create(prim::Constant);
    constant->output()->setType(c10::IntType::get());
    constant->i_(attr::value, input);
    graph_->prependNode(constant);
    inputs.push_back(constant->output());
  }

  void deleteOpNodeAndLists(Node* op_node) {
    // Collect the lists before we destroy op_node
    std::vector<Node*> lists;
    const auto& list_indices = getListIndices(op_node);
    lists.reserve(list_indices.size());
    for (const size_t list_idx : list_indices) {
      auto* list = op_node->input(list_idx)->node();
      lists.push_back(list);
    }

    GRAPH_UPDATE("Deleting\n", *op_node);
    op_node->destroy();
    for (auto* list : lists) {
      if (!list->hasUses()) {
        GRAPH_UPDATE("Deleting\n", *list);
        list->destroy();
      }
    }
  }

  bool replaceWithVariadicOp(Node* op_node) {
    if (!allListInputsAreValid(op_node)) {
      return false;
    }

    std::vector<Value*> inputs;
    size_t cur_idx = 0;
    std::vector<size_t> list_lens;
    for (const size_t list_idx : getListIndices(op_node)) {
      insertAllInputsBetween(inputs, op_node, cur_idx, list_idx);
      const auto list = op_node->input(list_idx)->node();
      const auto list_len = list->inputs().size();
      list_lens.push_back(list_len);
      insertAllInputsBetween(inputs, list, 0, list_len);
      cur_idx = list_idx + 1;
    }
    insertAllInputsBetween(inputs, op_node, cur_idx, op_node->inputs().size());

    // We insert these extra integers at the end of the argument list only if we
    // have more than one variadic list (the information is redundant when there
    // is only one list because the interpreter knows how many arguments there
    // are).
    if (list_lens.size() > 1) {
      for (const size_t list_len : list_lens) {
        insertIntegerInput(inputs, list_len);
      }
    }

    auto var_op_node = op_node->owningGraph()->create(variadic_op_, inputs);
    var_op_node->output()->setType(op_node->output()->type());
    GRAPH_UPDATE("Adding\n", *var_op_node);
    var_op_node->insertBefore(op_node);
    GRAPH_UPDATE("Replacing\n", *op_node, "with\n", *var_op_node);
    op_node->output()->replaceAllUsesWith(var_op_node->output());
    deleteOpNodeAndLists(op_node);
    return true;
  }

  std::shared_ptr<Graph> graph_;
  std::vector<Node*> op_nodes_;

  AliasDb alias_db_;

  NodeKind op_;
  NodeKind variadic_op_;

  std::unordered_map<std::string, std::vector<size_t>> schema_to_list_indices_;
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
    ConstantPooling(graph);
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

} // namespace torch::jit
