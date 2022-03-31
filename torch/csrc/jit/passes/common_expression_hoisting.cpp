#include <torch/csrc/jit/passes/common_expression_hoisting.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/jit_log.h>

#include <cstddef>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace {

struct CommonExpressionHoister {
  CommonExpressionHoister(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  bool run() {
    HoistCommonExpression(graph_->block());
    return changed_;
  }

  void HoistFromIfNode(Node* if_node) {
    Block* true_block = if_node->blocks()[0];
    Block* false_block = if_node->blocks()[1];
    // find common statements in the two subblocks

    auto true_block_nodes = std::unordered_set<Node*, HashNode, EqualNode>(
        true_block->nodes().begin(), true_block->nodes().end());
    for (auto it = false_block->nodes().begin();
         it != false_block->nodes().end();) {
      Node* false_b_node = *it;
      // node may be moved to a different block so advance iterator now
      ++it;

      auto matching_elem = true_block_nodes.find(false_b_node);
      if (matching_elem == true_block_nodes.end()) {
        continue;
      }
      Node* true_b_node = *matching_elem;

      // Check if a move to the front of the block is valid
      AliasDb& aliasDb = getOrCreateAliasDb();
      bool true_moveable = aliasDb.couldMoveBeforeTopologically(
          true_b_node, true_block->nodes().front());
      bool false_moveable = aliasDb.couldMoveBeforeTopologically(
          false_b_node, false_block->nodes().front());

      if (!true_moveable || !false_moveable) {
        continue;
      }

      bool did_move_true = aliasDb.moveBeforeTopologicallyValid(
          true_b_node, true_block->nodes().front());
      bool did_move_false = aliasDb.moveBeforeTopologicallyValid(
          false_b_node, false_block->nodes().front());

      TORCH_INTERNAL_ASSERT(
          did_move_true && did_move_false,
          "Wasn't able to move nodes to the beginning of the if blocks");

      // Even though we moved the node before the original first node in the
      // block, we might have also moved other nodes in front of the target
      // node. If it ended up at the very beginning of the if block, then it's
      // safe to move it outside.
      if (true_b_node != true_block->nodes().front() ||
          false_b_node != false_block->nodes().front()) {
        continue;
      }

      // Get all the uses of the output to delete and reinsert them
      // as the input would change, the HashNode value would also change.
      std::unordered_set<Node*> true_b_uses;
      for (Value* true_out : true_b_node->outputs()) {
        for (Use true_use : true_out->uses()) {
          if (true_use.user->owningBlock() == true_block) {
            // Make sure we are not accidentally adding stuff from subblocks
            true_b_uses.insert(true_use.user);
          }
        }
      }
      for (Node* uses_node : true_b_uses) {
        true_block_nodes.erase(uses_node);
      }

      // Now hoist the statement out of the block
      changed_ = true;
      false_b_node->moveBefore(if_node);

      true_b_node->replaceAllUsesWith(false_b_node);

      true_block_nodes.erase(true_b_node);
      true_block_nodes.insert(true_b_uses.cbegin(), true_b_uses.cend());
      true_b_node->destroy();
    }
  }

  void EliminateUnnecessaryIfOutputs(Node* if_node) {
    Block* true_block = if_node->blocks()[0];
    Block* false_block = if_node->blocks()[1];

    // fix up the if block outputs
    for (size_t i = 0; i < true_block->outputs().size();) {
      // Need to check both sides match to eliminate common if block outputs
      Value* true_block_output = true_block->outputs().at(i);
      Value* false_block_output = false_block->outputs().at(i);
      if (true_block_output != false_block_output) {
        i++;
        continue;
      }

      // We have a matching output, and can remove it from the block itself
      if_node->outputs().at(i)->replaceAllUsesWith(true_block_output);
      if_node->eraseOutput(i);
      true_block->eraseOutput(i);
      false_block->eraseOutput(i);
      changed_ = true;
    }

    // No need to test here if the IF block should be eliminated.
    // The DCE pass will determine that for us.
  }

  void HoistCommonExpression(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* node = *it;
      ++it;

      for (auto sub_block : node->blocks()) {
        HoistCommonExpression(sub_block);
      }

      if (node->kind() == prim::If) {
        HoistFromIfNode(node);
        EliminateUnnecessaryIfOutputs(node);
      }
    }
  }

  AliasDb& getOrCreateAliasDb() {
    if (!alias_db_) {
      alias_db_ = std::make_unique<AliasDb>(graph_);
    }

    return *alias_db_;
  }

 private:
  std::unique_ptr<AliasDb> alias_db_;
  std::shared_ptr<Graph> graph_;
  bool changed_ = false;
};
} // anonymous namespace
bool HoistCommonExpression(const std::shared_ptr<Graph>& graph) {
  // This moves common subexpressions from the two sides of an
  // if block out of the if block.

  GRAPH_DUMP("Before CEH", graph);
  CommonExpressionHoister ceh(graph);
  bool changed = ceh.run();
  if (changed) {
    GRAPH_DUMP("After CEH Changes", graph);
  }
  return changed;
}
} // namespace jit
} // namespace torch
