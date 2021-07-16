#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/restore_mutation.h>

namespace torch {
namespace jit {

bool MutationRemover::removeListMutation() {
  return RemoveListMutation(graph_->block());
}

bool MutationRemover::removeTensorMutation() {
  return RemoveTensorMutation(graph_->block());
}

bool MutationRemover::hasSideEffectOrAlias(Value* v, AliasDb* aliasDb) {
  // bail on nodes with side effects, blocks, or graph / graph inputs
  Node* n = v->node();
  bool unhandled_node = n->blocks().size() != 0 ||
      n->hasAttribute(attr::Subgraph) || n->hasSideEffects() ||
      (v->node()->kind() == prim::Param);

  // if the output isn't contained or alias by the inputs to its node, it's
  // unique
  return unhandled_node || aliasDb->mayContainAlias(v->node()->inputs(), v) ||
      (v->node()->kind() == prim::Param);
}

Node* MutationRemover::createSpecialMappedOp(Node* n) {
  WithInsertPoint guard(n);
  auto inputs = n->inputs();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Node* new_node;
  if (n->matches(
          "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)")) {
    new_node =
        graph_->insert(aten::full_like, {inputs.at(0), inputs.at(1)})->node();
  } else if (n->matches("aten::zero_(Tensor(a!) self) -> Tensor(a!)")) {
    new_node = graph_->insert(aten::zeros_like, {n->inputs().at(0)})->node();
  } else if (
      n->matches(
          "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)")) {
    // TODO: we should have normal_like operator
    // normal(float mean, float std, int[] size, *, Generator? generator=None,
    // ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None) -> Tensor
    auto size = graph_->insert(aten::size, {n->inputs().at(0)});
    auto dtype = graph_->insert(prim::dtype, {n->inputs().at(0)});
    auto layout = graph_->insert(prim::layout, {n->inputs().at(0)});
    auto device = graph_->insert(prim::device, {n->inputs().at(0)});
    auto pin_memory = graph_->insert(aten::is_pinned, {n->inputs().at(0)});
    auto generator = graph_->insertConstant(IValue());
    new_node = graph_->insertNode(graph_->create(
        aten::normal,
        {n->inputs().at(1),
         n->inputs().at(2),
         size,
         generator,
         dtype,
         layout,
         device,
         pin_memory}));
  } else {
    TORCH_INTERNAL_ASSERT(false);
  }
  new_node->copyMetadata(n);
  new_node->output()->setType(n->output()->type());
  return new_node;
}

bool MutationRemover::listMutationFollowingListConstruct(Node* n) {
  return (
      (n->kind() == aten::append ||
       (n->kind() == aten::insert &&
        n->inputs().at(1)->node()->kind() == prim::Constant)) &&
      n->inputs().at(0)->node()->kind() == prim::ListConstruct);
}

bool MutationRemover::tryMakeCreationAndMutationAtomic(
    Value* mutated_value,
    Node* mutating_op) {
  // We can only remove mutation to values that are unique aliases in the
  // graph. if x = y[0] or y = self.y, then removing the mutation could
  // change observable semantics
  if (hasSideEffectOrAlias(mutated_value, getOrCreateAliasDb())) {
    return false;
  }

  // In order to safely remove a mutation, the creation of a tensor and its
  // subsequent mutation need to be one atomic operation
  return getOrCreateAliasDb()->moveBeforeTopologicallyValid(
      mutated_value->node(), mutating_op);
}

bool MutationRemover::tryMakeUnaliasedIfOutputAndMutationAtomic(
    Value* mutated_value,
    Node* mutating_op) {
  // if cond:
  //    x = op()
  // else:
  //    x = op()
  // x = add_(1)
  // if x in both blocks have no other uses and are unaliased in the graph,
  // and we make the if node and the mutation atomic,
  // then removing mutation add_ does not change observable semantics

  if (mutated_value->node()->kind() != prim::If) {
    return false;
  }

  auto if_node = mutated_value->node();
  auto offset = mutated_value->offset();
  auto true_value = if_node->blocks().at(0)->outputs().at(offset);
  auto false_value = if_node->blocks().at(1)->outputs().at(offset);

  if (true_value->uses().size() > 1 || false_value->uses().size() > 1) {
    return false;
  }

  if (hasSideEffectOrAlias(true_value, getOrCreateAliasDb()) ||
      hasSideEffectOrAlias(false_value, getOrCreateAliasDb())) {
    return false;
  }

  return getOrCreateAliasDb()->moveBeforeTopologicallyValid(
      if_node, mutating_op);
}

bool MutationRemover::RemoveListMutation(Block* block) {
  bool changed = false;
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto* node = *it;
    it++;

    for (Block* sub_block : node->blocks()) {
      changed |= RemoveListMutation(sub_block);
    }

    if (!listMutationFollowingListConstruct(node)) {
      continue;
    }

    Value* mutated_value = node->inputs().at(0);
    if (!tryMakeCreationAndMutationAtomic(mutated_value, node)) {
      continue;
    }

    changed = true;

    // We rewrite something like:
    // x = {v0}
    // x.append(v1) (or x.insert(0, v1))
    // to:
    // x = {v0, v1} (or x = {v1, v0})
    // We can remove x.append from the the alias db list of writes.
    // All other aliasing properties remain valid.
    Node* list_construct = mutated_value->node();
    switch (node->kind()) {
      case aten::append:
        list_construct->addInput(node->inputs().at(1));
        break;
      case aten::insert: {
        int pos = toIValue(node->inputs().at(1))->toInt();
        int size = list_construct->inputs().size();
        // insert to neg position equals insert to std::max(pos+size, 0)
        if (pos < 0) {
          pos = std::max(pos + size, 0);
        }
        // insert beyond current list length is the same as append
        pos = std::min(pos, size);
        list_construct->insertInput(pos, node->inputs().at(2));
        break;
      }
      default:
        TORCH_INTERNAL_ASSERT(false);
    }

    // process use-chain and aliasing of node output
    bool has_output = (node->outputs().size() > 0);
    if (has_output) {
      node->output()->replaceAllUsesWith(mutated_value);
      getOrCreateAliasDb()->writeIndex_->erase(node);
    }

    node->destroy();

    // TODO: don't strictly need to reset write cache, evaluate on models
    getOrCreateAliasDb()->buildWrittenToLocationsIndex();
  }

  return changed;
}

bool MutationRemover::RemoveTensorMutation(Block* block) {
  bool changed = false;
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    auto* node = *it;
    it++;

    for (Block* sub_block : node->blocks()) {
      changed |= RemoveTensorMutation(sub_block);
    }

    if (mutation_filter_) {
      const auto& mutation_filter = *mutation_filter_;
      if (!mutation_filter(node)) {
        continue;
      }
    }

    // TODO: out op variants
    if (!inplaceOpVariant(node)) {
      continue;
    }

    Value* mutated_value = node->inputs().at(0);
    if (!tryMakeCreationAndMutationAtomic(mutated_value, node) &&
        !tryMakeUnaliasedIfOutputAndMutationAtomic(mutated_value, node)) {
      continue;
    }

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Node* new_node;
    if (isSpecialMappedOp(node)) {
      new_node = createSpecialMappedOp(node);
    } else {
      auto schema_name = node->schema().name();
      auto new_schema = schema_name.substr(0, schema_name.size() - 1);
      new_node = graph_->create(Symbol::fromQualString(new_schema), 1);
      new_node->copyMetadata(node);
      new_node->insertBefore(node);
      for (Value* input : node->inputs()) {
        new_node->addInput(input);
      }
      new_node->output()->setType(node->output()->type());

      // weird case where there is an inplace op and an equivalent functional op
      // of the same symbol, but they have different schemas
      if (!new_node->maybeOperator()) {
        new_node->destroy();
        continue;
      }
    }

    changed = true;
    mutated_value->replaceAllUsesAfterNodeWith(node, new_node->output());
    node->output()->replaceAllUsesWith(new_node->output());

    // We rewrite something like:
    // x = torch.zeros()
    // x.add_(1)
    // x.add_(2)
    // to:
    // x = torch.zeros()
    // x0 = x.add(1)
    // x0.add_(2)
    // For the remainder of the function, x0 will have the
    // same aliasing relationships as the original x.
    // To avoid rebuilding the entire alias db, we can replace
    // the memory DAG element of x with x0.
    getOrCreateAliasDb()->replaceWithNewValue(
        mutated_value, new_node->output());

    // it is an invariant that all mutable types have an element in the memory
    // DAG so we must regive x an alias db element. We have already verified
    // that the mutated value is a fresh alias with a single use.
    getOrCreateAliasDb()->createValue(mutated_value);

    // We must erase the destroyed node from the AliasDb lists of writes
    getOrCreateAliasDb()->writeIndex_->erase(node);
    node->destroy();

    // now that we have removed a mutating op, the write cache is stale
    // TODO: don't strictly need to reset write cache, evaluate on models
    getOrCreateAliasDb()->buildWrittenToLocationsIndex();
  }

  return changed;
}

bool MutationRemover::inplaceOpVariant(Node* n) {
  if (!n->kind().is_aten()) {
    return false;
  }

  if (isSpecialMappedOp(n)) {
    return true;
  }

  auto name = n->schema().name();
  bool inplace_op = name.at(name.size() - 1) == '_';
  if (!inplace_op) {
    return false;
  }

  // needs to have alias analysis by schema
  auto op = n->maybeOperator();
  if (!op) {
    return false;
  }
  if (op->aliasAnalysisKind() != AliasAnalysisKind::FROM_SCHEMA) {
    return false;
  }

  // all inplace ops at time of writing have a single input that is mutated
  // and returned. check that this is true, anything else could have strange
  // semantics,
  if (n->outputs().size() != 1 || n->inputs().size() == 0) {
    return false;
  }
  auto inputs = n->inputs();
  if (!getOrCreateAliasDb()->writesToAlias(n, {inputs.at(0)}) ||
      getOrCreateAliasDb()->writesToAlias(
          n, {inputs.slice(1).begin(), inputs.slice(1).end()})) {
    return false;
  }

  auto new_schema = name.substr(0, name.size() - 1);
  return getAllOperatorsFor(Symbol::fromQualString(new_schema)).size() != 0;
}

bool RemoveListMutation(const std::shared_ptr<Graph>& graph) {
  MutationRemover mr(graph);
  return mr.removeListMutation();
}

bool RemoveTensorMutation(
    const std::shared_ptr<Graph>& graph,
    c10::optional<std::function<bool(Node*)>> mutation_filter) {
  MutationRemover mr(graph, std::move(mutation_filter));
  return mr.removeTensorMutation();
}

static const std::unordered_set<Symbol> activation_ops = []() {
  std::unordered_set<Symbol> target_ops;
  for (const auto& iter : activation_type_promotion_mapping) {
    std::string name = std::string(iter.first.toQualString()) + "_";
    target_ops.insert(Symbol::fromQualString(name));
  }
  return target_ops;
}();

bool InplaceToFunctionalActivation(const std::shared_ptr<Graph>& graph) {
  return RemoveTensorMutation(graph, [](Node* node) {
    return activation_ops.count(node->kind()) != 0;
  });
}

} // namespace jit
} // namespace torch
