#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

class FunctionalActivationRewriter {
 public:
  FunctionalActivationRewriter(std::shared_ptr<Graph> graph);

  bool FunctionalToInplaceActivation(Block* block);

 private:
  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  std::unordered_set<Symbol> functional_activation_ops_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::shared_ptr<Graph> graph_;
};

// Replaces in-place aten activation ops with their functional equivalents
TORCH_API bool InplaceToFunctionalActivation(
    const std::shared_ptr<Graph>& graph);

// Replaces functional aten activation ops with their in-place equivalents
TORCH_API bool FunctionalToInplaceActivation(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
