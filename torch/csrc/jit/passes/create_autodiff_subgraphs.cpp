#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch {
namespace jit {

namespace {

class SubgraphSlicer {
 public:
  SubgraphSlicer(
      Block* block,
      std::shared_ptr<Graph> graph,
      size_t minSubgraphSize)
      : block_(block),
        graph_(std::move(graph)),
        minSubgraphSize_(minSubgraphSize) {}

  void run(std::vector<Node*>& diffGraphs) {
    // We need to run the slicer multiple times in order to get all merge
    // opportunities. This is because moveBeforeTopologicalValid may reorder
    // nodes to be AFTER the current iteration point. In order to properly
    // consider those nodes for merging, we need run the pass until no changes
    // have been made.
    //
    // Example:
    //   c = f(a, b)
    //   d = f(c)
    //   e = f(d)  <- iter is here, moving upward
    // After c.moveBeforeTopologicallyValid(e), we have:
    //   c = f(a, b)
    //   e = f(d)  <- iter still here
    //   d = f(c)  <- this was node moved on the other side.
    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      AliasDb aliasDb(graph_);
      for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();) {
        bool changed;
        std::tie(it, changed) = scanNode(*it, aliasDb);
        any_changed |= changed;
      }
    }

    // Done constructing subgraphs. Do some post-processing cleanup:
    // 1. Run CSE to delete redundanet constant nodes.
    // 2. We may need to re-inline ones that are too small.
    auto curNode = *block_->nodes().rbegin();
    while (curNode != *block_->nodes().rend()) {
      for (auto subBlock : curNode->blocks()) {
        SubgraphSlicer(subBlock, graph_, minSubgraphSize_).run(diffGraphs);
      }

      // Save the previous node, since we might delete `curNode` in next block
      auto prevNode = curNode->prev();
      if (curNode->kind() == prim::DifferentiableGraph) {
        // Inlining nodes may cause some subexpression to come back in the
        // subgraphs (for example, copying constants in repeatedly will generate
        // redundant prim::Constants). Run CSE to clean them up.
        EliminateCommonSubexpression(curNode->g(attr::Subgraph));

        if (!inlineIfTooSmall(curNode)) {
          diffGraphs.push_back(curNode);
        }
      }
      curNode = prevNode;
    }
    // Run CSE one more time to eliminate duplicates that may have occurred
    // while re-inlining subgraphs.
    EliminateCommonSubexpression(graph_);
  }

 private:
  // Inline this node's group subgraph into the outer graph if it's smaller
  // than the specified minimum size.
  //
  // Returns true if an inlining has occurred, false otherwise.
  bool inlineIfTooSmall(Node* n) {
    AT_ASSERT(n->kind() == prim::DifferentiableGraph);
    auto subgraph = SubgraphUtils::getSubgraph(n);
    size_t i = 0;
    for (auto it = subgraph->nodes().begin(); it != subgraph->nodes().end();
         ++it) {
      if (++i >= minSubgraphSize_) {
        return false;
      }
    }

    SubgraphUtils::unmergeSubgraph(n);
    return true;
  }

  value_list sortReverseTopological(ArrayRef<Value*> inputs) {
    value_list result;
    for (auto i : inputs) {
      if (i->node()->owningBlock() == block_) {
        result.push_back(i);
      }
    }
    // Sort in reverse topological order
    std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
      return a->node()->isAfter(b->node());
    });
    return result;
  }

  bool shouldConsiderForMerge(Node* node) {
    // if we're already in the process of merging
    if (node->kind() == prim::DifferentiableGraph) {
      return true;
    }
    if (node->kind() == prim::Constant) {
      return false;
    }
    return isDifferentiable(node);
  }

  std::pair<graph_node_list::iterator, bool> scanNode(
      Node* consumer,
      AliasDb& aliasDb) {
    if (shouldConsiderForMerge(consumer)) {
      if (consumer->kind() != prim::DifferentiableGraph) {
        consumer = SubgraphUtils::createSingletonSubgraph(
            consumer, prim::DifferentiableGraph);
      }
      auto inputs = sortReverseTopological(consumer->inputs());
      for (auto input : inputs) {
        if (auto group = tryMerge(consumer, input->node(), aliasDb)) {
          // we successfully merged, so the new group's `inputs` may have
          // changed. So rescan the new group for more merging opportunities.
          return std::make_pair(group.value()->reverseIterator(), true);
        }
      }
    }

    return std::make_pair(++consumer->reverseIterator(), false);
  }

  // Try to merge `producer` into `consumer`. If successful, this destroys
  // `producer` and returns the `consumer` group.
  c10::optional<Node*> tryMerge(
      Node* consumer,
      Node* producer,
      AliasDb& aliasDb) {
    AT_ASSERT(consumer->kind() == prim::DifferentiableGraph);
    bool canMerge = shouldConsiderForMerge(producer) &&
        aliasDb.moveBeforeTopologicallyValid(producer, consumer);

    if (!canMerge) {
      return c10::nullopt;
    }

    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);

    return consumer;
  }

  Block* block_;
  std::shared_ptr<Graph> graph_;
  size_t minSubgraphSize_;
};
} // anonymous namespace

std::vector<Node*> CreateAutodiffSubgraphs(
    const std::shared_ptr<Graph>& graph,
    size_t threshold) {
  std::vector<Node*> diff_nodes;
  SubgraphSlicer(graph->block(), graph, threshold).run(diff_nodes);
  return diff_nodes;
}
} // namespace jit
} // namespace torch
