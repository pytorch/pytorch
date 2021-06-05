#include <torch/csrc/jit/passes/common_expression_hoisting.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/jit_log.h>

#include <cstddef>
#include <unordered_set>

namespace torch {
namespace jit {
namespace {

bool HoistFromIfNode(Node* if_node, AliasDb& aliasDb) {
  Block* true_block = if_node->blocks()[0];
  Block* false_block = if_node->blocks()[1];
  // find common statements in the two subblocks
  bool block_modified = false;

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

    // Check if there are any mutatiors for the output blocks
    // Technically you only need to check for it in the IF block
    if (aliasDb.hasOutputWriters(false_b_node) ||
        aliasDb.hasOutputWriters(true_b_node)) {
      continue;
    }

    // Check if a move to the front of the block is valid
    // If both of the moves are valid, then we know we can move the item out of
    // the if blocks entirely.
    bool true_moveable = aliasDb.couldMoveBeforeTopologically(
        true_b_node, true_block->nodes().front());
    bool false_moveable = aliasDb.couldMoveBeforeTopologically(
        false_b_node, false_block->nodes().front());

    if (!true_moveable || !false_moveable) {
      continue;
    }
    // Now do the hoist”
    block_modified = true;
    false_b_node->moveBefore(if_node);
    true_b_node->replaceAllUsesWith(false_b_node);

    // fix up the if block outputs

    true_block_nodes.erase(true_b_node);
    true_b_node->destroy();
  }

  for (size_t i = 0; i < true_block->outputs().size();) {
    // Need to check both sides match to eliminate common if block outputs
    Value* true_block_output = true_block->outputs().at(i);
    Value* false_block_output = false_block->outputs().at(i);
    if (true_block_output != false_block_output ||
        aliasDb.hasWriters(true_block_output)) {
      i++;
      continue;
    }
    // We have a matching output, and can hoist out of if block
    if_node->outputs().at(i)->replaceAllUsesWith(true_block_output);
    if_node->eraseOutput(i);
    true_block->eraseOutput(i);
    false_block->eraseOutput(i);
  }
  // We will let DCE elimiate the if block if it should be eliminated.
  return block_modified;
}

bool HoistCommonExpression(Block* block, AliasDb& aliasDb) {
  bool var_hoisted = false;
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* node = *it;
    ++it;

    for (auto sub_block : node->blocks()) {
      HoistCommonExpression(sub_block, aliasDb);
    }

    if (node->kind() != prim::If) {
      continue;
    }
    // Note that they will not have the same outputs
    var_hoisted = HoistFromIfNode(node, aliasDb);
  }
  return var_hoisted;
}
} // anonymous namespace
bool HoistCommonExpression(const std::shared_ptr<Graph>& graph) {
  // This moves common subexpressions from the two sides of an
  // if block out of the if block.

  GRAPH_DUMP("Before CSE", graph);
  AliasDb aliasDb(graph);
  return HoistCommonExpression(graph->block(), aliasDb);
}
} // namespace jit
} // namespace torch
