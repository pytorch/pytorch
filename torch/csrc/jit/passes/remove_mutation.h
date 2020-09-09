#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

struct MutationRemover {
  MutationRemover(const std::shared_ptr<Graph>& graph)
      : aliasDb_(nullptr), graph_(graph) {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
  }

  void removeListMutation();

  void removeTensorMutation();

  bool isSpecialMappedOp(Node* n) {
    return n->matches("aten::zero_(Tensor(a!) self) -> Tensor(a!)") ||
        n->matches(
            "aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)");
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
  bool listAppendFollowingListConstruct(Node* n);
  bool tryMakeCreationAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op);
  bool tryMakeUnaliasedIfOutputAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op);
  void RemoveListMutation(Block* block);
  void RemoveTensorMutation(Block* block);

 private:
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::shared_ptr<Graph> graph_;
};

// Replaces in-place aten ops with their functional equivalents
// when it can be proven that this does not change graph semantics
TORCH_API void RemoveListMutation(const std::shared_ptr<Graph>& graph);

// Removes list mutation with functional equivalents
TORCH_API void RemoveTensorMutation(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
