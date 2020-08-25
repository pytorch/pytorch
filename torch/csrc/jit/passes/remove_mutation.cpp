#include <torch/csrc/jit/passes/remove_mutation.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

struct MutationRemover {
  MutationRemover(const std::shared_ptr<Graph>& graph)
      : aliasDb_(nullptr), graph_(graph) {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
  }

  void removeListMutation() {
    RemoveListMutation(graph_->block());
  }

  void removeTensorMutation() {
    RemoveTensorMutation(graph_->block());
  }

 private:
  bool newMemoryLocation(Value* v) {
    // bail on nodes with side effects, blocks, or graph / graph inputs
    Node* n = v->node();
    bool unhandled_node = n->blocks().size() != 0 ||
        n->hasAttribute(attr::Subgraph) || n->hasSideEffects() ||
        (v->node()->kind() == prim::Param);

    // if the output isn't contained or alias by the inputs to its node, it's
    // unique
    return !unhandled_node &&
        !aliasDb_->mayContainAlias(v->node()->inputs(), v) &&
        !(v->node()->kind() == prim::Param);
  }

  bool isSpecialMappedOp(Node* n) {
    return n->matches("aten::zero_(Tensor(a!) self) -> Tensor(a!)") ||
        n->matches(
            "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)");
  }

  Node* createSpecialMappedOp(Node* n) {
    WithInsertPoint guard(n);
    auto inputs = n->inputs();
    Node* new_node;
    if (n->matches(
            "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)")) {
      new_node =
          graph_->insert(aten::full_like, {inputs.at(0), inputs.at(1)})->node();
    } else if (n->matches("aten::zero_(Tensor(a!) self) -> Tensor(a!)")) {
      new_node = graph_->insert(aten::zeros_like, {n->inputs().at(0)})->node();
    } else {
      TORCH_INTERNAL_ASSERT(false);
    }
    new_node->copyMetadata(n);
    new_node->output()->setType(n->output()->type());
    return new_node;
  }

  bool inplaceOpVariant(Node* n) {
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
    if (!aliasDb_->writesToAlias(n, {inputs.at(0)}) ||
        aliasDb_->writesToAlias(
            n, {inputs.slice(1).begin(), inputs.slice(1).end()})) {
      return false;
    }

    auto new_schema = name.substr(0, name.size() - 1);
    return getAllOperatorsFor(Symbol::fromQualString(new_schema)).size() != 0;
  }

  bool listAppendFollowingListConstruct(Node* n) {
    return n->kind() == aten::append &&
        n->inputs().at(0)->node()->kind() == prim::ListConstruct;
  }

  bool tryMakeCreationAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op) {
    // We can only remove mutation to values that are unique aliases in the
    // graph. if x = y[0] or y = self.y, then removing the mutation could
    // change observable semantics
    if (!newMemoryLocation(mutated_value)) {
      return false;
    }

    // In order to safely remove a mutation, the creation of a tensor and its
    // subsequent mutation need to be one atomic operation
    return aliasDb_->moveBeforeTopologicallyValid(
        mutated_value->node(), mutating_op);
  }

  bool tryMakeUnaliasedIfOutputAndMutationAtomic(
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

    if (!newMemoryLocation(true_value) || !newMemoryLocation(false_value)) {
      return false;
    }

    return aliasDb_->moveBeforeTopologicallyValid(if_node, mutating_op);
  }

  void RemoveListMutation(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      auto* node = *it;
      it++;

      for (Block* sub_block : node->blocks()) {
        RemoveListMutation(sub_block);
      }

      if (!listAppendFollowingListConstruct(node)) {
        continue;
      }

      Value* mutated_value = node->inputs().at(0);
      if (!tryMakeCreationAndMutationAtomic(mutated_value, node)) {
        continue;
      }

      // We rewrite something like:
      // x = {v0}
      // x.append(v1)
      // to:
      // x = {v0, v1}
      // We can remove x.append from the the alias db list of writes.
      // All other aliasing properties remain valid.
      Node* list_construct = mutated_value->node();
      list_construct->addInput(node->inputs().at(1));
      node->output()->replaceAllUsesWith(mutated_value);
      aliasDb_->writeIndex_->erase(node);
      node->destroy();

      // TODO: don't strictly need to reset write cache, evaluate on models
      aliasDb_->writtenToLocationsIndex_ =
          aliasDb_->buildWrittenToLocationsIndex();
    }
  }

  void RemoveTensorMutation(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      auto* node = *it;
      it++;

      for (Block* sub_block : node->blocks()) {
        RemoveTensorMutation(sub_block);
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
      }

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
      // the memory dag element of x with x0.
      aliasDb_->replaceWithNewValue(mutated_value, new_node->output());

      // it is an invariant that all mutable types have an element in the memory
      // dag so we must regive x an alias db element. We have already verified
      // that the mutated value is a fresh alias with a single use.
      aliasDb_->createValue(mutated_value);

      // We must erase the destroyed node from the AliasDb lists of writes
      aliasDb_->writeIndex_->erase(node);
      node->destroy();

      // now that we have removed a mutating op, the write cache is stale
      // TODO: don't strictly need to reset write cache, evaluate on models
      aliasDb_->writtenToLocationsIndex_ =
          aliasDb_->buildWrittenToLocationsIndex();
    }
  }

 private:
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::shared_ptr<Graph> graph_;
};

void RemoveListMutation(const std::shared_ptr<Graph>& graph) {
  MutationRemover mr(graph);
  mr.removeListMutation();
}

void RemoveTensorMutation(const std::shared_ptr<Graph>& graph) {
  MutationRemover mr(graph);
  mr.removeTensorMutation();
}

} // namespace jit
} // namespace torch
