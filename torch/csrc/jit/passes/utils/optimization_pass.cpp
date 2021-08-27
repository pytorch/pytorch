
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/utils/memory.h>
#include <vector>
#include "ATen/core/functional.h"
#include "ATen/core/interned_strings.h"
#include "jit/jit_log.h"

#include <torch/csrc/jit/passes/utils/optimization_pass.h>



namespace torch {
namespace jit {

bool OptimizationPass::run() {
  handleBlockAndSubblocks(graph_->block());
  return graph_modified;
}

void OptimizationPass::handleBlockAndSubblocks(Block* block) {
  for (auto node : block->nodes()) {
    for (Block* block : node->blocks()) {
      handleBlockAndSubblocks(block);
    }
  }
  handleBlock(block);
}

AliasDb* OptimizationPass::getAliasDb() {
  if (!aliasDb_) {
    aliasDb_ = std::make_unique<AliasDb>(graph_);
  }
  return aliasDb_.get();
}

} // namespace jit
} // namespace torch