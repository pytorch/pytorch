#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

struct FunctionalToInplaceRewriterFlags {
  bool transform;
  bool check_shape;
  bool check_dtype;
};

class FunctionalToInplaceRewriter {
 public:
  FunctionalToInplaceRewriter(
      std::shared_ptr<Graph> graph,
      c10::optional<std::function<FunctionalToInplaceRewriterFlags(Node*)>>
          node_filter = c10::nullopt);

  bool FunctionalToInplace(Block* block);

 private:
  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  bool CanBeInplace(Node* node);

  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::shared_ptr<Graph> graph_;
  c10::optional<std::function<FunctionalToInplaceRewriterFlags(Node*)>>
      node_filter_;
};

// A common application scenario is to apply InplaceToFunctionalActivation
// before some JIT optimization passes, so that those passes are less
// constrained by in-place ops. After those passes are done, we can call
// FunctionalToInplaceActivation to recover in-place activation ops,
// so that we won't lose the performance benefit coming from memory reduction.

// Replaces in-place aten activation ops with their functional equivalents
TORCH_API bool InplaceToFunctionalActivation(
    const std::shared_ptr<Graph>& graph);

// Replaces functional aten activation ops with their in-place equivalents
TORCH_API bool FunctionalToInplaceActivation(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
