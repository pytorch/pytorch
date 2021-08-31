
#pragma once

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>


/** \brief A Class that holds common methods for an optimization pass.
 *

*/

namespace torch {
namespace jit {

class OptimizationPass{
 public:
  explicit OptimizationPass(const std::shared_ptr<Graph>& graph)
      : graph_(std::move(graph)) {}

  bool run();
  // Runs the relevant optimization. Returns if the graph was modified.

  protected:
  void handleBlockAndSubblocks(Block* block);
  // Handles the current block and recursively handle subblocks

  virtual void handleBlock(Block* block) = 0;
  // Per Block Logic that subclasses implement

  AliasDb* getAliasDb();
  // Get AliasDB. Creates a new instance if we don't already have one

  const std::shared_ptr<Graph> graph_;
  bool graph_modified = false;

  private:
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
};

} // namespace jit
} // namespace torch