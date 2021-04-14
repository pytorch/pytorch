#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

struct MutationRemover {
  MutationRemover(
      std::shared_ptr<Graph> graph,
      c10::optional<std::function<bool(Node*)>> mutation_filter = c10::nullopt)
      : aliasDb_(nullptr), graph_(std::move(graph)) {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
    mutation_filter_ = mutation_filter;
  }

  // return true if graph is modified
  bool removeListMutation();

  // return true if graph is modified
  bool removeTensorMutation();

  bool isSpecialMappedOp(Node* n) {
    return n->matches("aten::zero_(Tensor(a!) self) -> Tensor(a!)") ||
        n->matches(
            "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)") ||
        n->matches(
            "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)");
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

 private:
  bool newMemoryLocation(Value* v);
  Node* createSpecialMappedOp(Node* n);
  bool listMutationFollowingListConstruct(Node* n);
  bool tryMakeCreationAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op);
  bool tryMakeUnaliasedIfOutputAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op);
  // return true if graph is modified
  bool RemoveListMutation(Block* block);
  // return true if graph is modified
  bool RemoveTensorMutation(Block* block);

  c10::optional<std::function<bool(Node*)>> mutation_filter_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::shared_ptr<Graph> graph_;
};

// Removes list mutation with functional equivalents
// return true if graph is modified
TORCH_API bool RemoveListMutation(const std::shared_ptr<Graph>& graph);

// Replaces in-place aten ops with their functional equivalents
// when it can be proven that this does not change graph semantics
// if `mutation_filter` is present, the pass will only attempt to
// remove mutation on nodes which return true for the filter
// return true if graph is modified
TORCH_API bool RemoveTensorMutation(
    const std::shared_ptr<Graph>& graph,
    c10::optional<std::function<bool(Node*)>> mutation_filter = c10::nullopt);

} // namespace jit
} // namespace torch
