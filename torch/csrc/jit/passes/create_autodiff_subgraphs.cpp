#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/autodiff.h>

namespace torch {
namespace jit {

namespace {

std::vector<c10::optional<const Use>> gatherLastUses(
    at::ArrayRef<Value*> values) {
  return fmap(values, [&](Value* v) -> c10::optional<const Use> {
    return firstOrLastUse(v, /*find_first*/ false);
  });
}

struct ValueMapper {
  ValueMapper(Node* n, AliasDb& db, size_t subgraph_num_outputs) {
    last_uses_ = gatherLastUses(n->outputs());
    subgraph_num_outputs_ = subgraph_num_outputs;
    WithInsertPoint guard(n);
    auto g = n->owningGraph();
    placeholder_node_ = g->insertNode(g->create(prim::Uninitialized, 0));
    for (size_t i = 0; i < n->outputs().size(); ++i) {
      Value* existing = n->outputs().at(i);
      Value* new_value =
          placeholder_node_->insertOutput(i)->copyMetadata(n->outputs().at(i));
      db.replaceWithNewValue(existing, new_value);
    }
  }

  bool usesEqual(const Use& a, const Use& b) {
    return a.user == b.user && a.offset == b.offset;
  }

  void copyAliasing(Node* merged_node, AliasDb& db) {
    auto num_outputs = merged_node->outputs().size();
    auto new_outputs = merged_node->outputs().slice(
        subgraph_num_outputs_, num_outputs - subgraph_num_outputs_);
    for (Value* v : new_outputs) {
      auto maybe_last_use = firstOrLastUse(v, /*find_first*/ false);
      if (!maybe_last_use) {
        continue;
      }
      const Use last_use = *maybe_last_use;

      size_t i = 0;
      while (i < last_uses_.size() && last_uses_.at(i).has_value() &&
             !usesEqual(*last_uses_.at(i), last_use)) {
        ++i;
      }
      TORCH_INTERNAL_ASSERT(i != last_uses_.size());
      db.replaceWithNewValue(placeholder_node_->outputs().at(i), v);
    }
    placeholder_node_->destroy();
  }

  std::vector<c10::optional<const Use>> last_uses_;
  size_t subgraph_num_outputs_;
  Node* placeholder_node_;
};

struct WorkBlock : public std::pair<Node*, Node*> {
  using pair::pair;

  Node* begin() {
    return this->first;
  }
  Node* end() {
    return this->second;
  }
};

class SubgraphPostProcessor {
 public:
  SubgraphPostProcessor(size_t minSubgraphSize, std::vector<Node*>& diff_nodes)
      : minSubgraphSize_(minSubgraphSize), diff_nodes_(diff_nodes) {}

  void run(Block* block) {
    auto curNode = *block->nodes().rbegin();
    while (curNode != *block->nodes().rend()) {
      // Save the previous node, since we might delete `curNode` in next block
      auto prevNode = curNode->prev();
      if (curNode->kind() == prim::DifferentiableGraph) {
        // Inlining nodes may cause some subexpression to come back in the
        // subgraphs (for example, copying constants in repeatedly will generate
        // redundant prim::Constants). Run CSE to clean them up.
        EliminateCommonSubexpression(curNode->g(attr::Subgraph));

        if (!inlineIfTooSmall(curNode)) {
          diff_nodes_.push_back(curNode);
        }
      }
      curNode = prevNode;
    }

    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        run(b);
      }
    }
  }

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
      // constants are not interpreted as instructions, ignore them
      i += it->kind() != prim::Constant;
      if (i >= minSubgraphSize_) {
        return false;
      }
    }

    SubgraphUtils::unmergeSubgraph(n);
    return true;
  }

  size_t minSubgraphSize_;
  std::vector<Node*>& diff_nodes_;
};

class SubgraphSlicer {
 public:
  SubgraphSlicer(
      Block* block,
      std::shared_ptr<Graph> graph,
      size_t minSubgraphSize,
      AliasDb& aliasDb)
      : block_(block),
        graph_(std::move(graph)),
        minSubgraphSize_(minSubgraphSize),
        aliasDb_(aliasDb) {}

  void run() {
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

    // see [workblocks]
    auto workblocks = buildWorkBlocks();
    for (auto& workblock : workblocks) {
      bool any_changed = true;
      while (any_changed) {
        any_changed = false;
        for (auto it = workblock.end()->reverseIterator();
             it != workblock.begin()->reverseIterator();) {
          bool changed;
          std::tie(it, changed) = scanNode(*it);
          any_changed |= changed;
        }
      }
    }

    // Construct Subgraphs Recursively
    // Creating autodiff subgraphs preserves the alias db, but it is difficult
    // to preserve the alias db when removing them, so we wait to do post
    // processing until all subgraphs are created
    for (Node* n : block_->nodes()) {
      for (auto subBlock : n->blocks()) {
        SubgraphSlicer(subBlock, graph_, minSubgraphSize_, aliasDb_).run();
      }
    }
  }

 private:
  std::vector<WorkBlock> buildWorkBlocks() {
    // [workblocks]
    // the IR has many nodes which can never be reordered around, such as a
    // prim::Bailout. if a node N is surrounded by two nodes which cannot be
    // reordered, A and B, then a differentiable subgraph that is created from N
    // can only contain nodes from (A, B) The nodes from A to B represent one
    // work block for the subgraph slicer to work on. By creating these up
    // front, we avoid retraversing the whole graph block any time scanNode
    // returns, and we can also avoid attempting to create differentiable
    // subgraphs in work blocks that do not contain a # of differentiable nodes
    // >= minSubgraphSize_

    Node* end_bound_node = block_->return_node();
    Node* curr = end_bound_node->prev();

    std::vector<WorkBlock> worklist;
    size_t differentiable_nodes = 0;

    while (curr != block_->param_node()) {
      differentiable_nodes += shouldConsiderForMerge(curr);

      // cannot reorder around side effectful nodes
      if (curr->hasSideEffects()) {
        // not enough differentiable nodes to create a differentiable subgraph
        if (differentiable_nodes >= minSubgraphSize_) {
          worklist.emplace_back(curr, end_bound_node);
        }
        differentiable_nodes = 0;
        end_bound_node = curr;
      }
      curr = curr->prev();
    }

    if (differentiable_nodes >= minSubgraphSize_) {
      worklist.emplace_back(curr, end_bound_node);
    }

    return worklist;
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

  std::pair<graph_node_list::iterator, bool> scanNode(Node* consumer) {
    if (shouldConsiderForMerge(consumer)) {
      if (consumer->kind() != prim::DifferentiableGraph) {
        // ValueMapper preserves the aliasing information of the node's outputs
        ValueMapper vm(consumer, aliasDb_, 0);
        consumer = SubgraphUtils::createSingletonSubgraph(
            consumer, prim::DifferentiableGraph);
        vm.copyAliasing(consumer, aliasDb_);
      }
      auto inputs = sortReverseTopological(consumer->inputs());
      for (auto input : inputs) {
        if (auto group = tryMerge(consumer, input->node())) {
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
  c10::optional<Node*> tryMerge(Node* consumer, Node* producer) {
    AT_ASSERT(consumer->kind() == prim::DifferentiableGraph);
    bool canMerge = shouldConsiderForMerge(producer) &&
        aliasDb_.moveBeforeTopologicallyValid(producer, consumer);

    if (!canMerge) {
      return c10::nullopt;
    }

    ValueMapper vm(producer, aliasDb_, consumer->outputs().size());
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
    vm.copyAliasing(consumer, aliasDb_);

    return consumer;
  }

  Block* block_;
  std::shared_ptr<Graph> graph_;
  size_t minSubgraphSize_;
  AliasDb& aliasDb_;
};
} // anonymous namespace

std::vector<Node*> CreateAutodiffSubgraphs(
    const std::shared_ptr<Graph>& graph,
    size_t threshold) {
  std::vector<Node*> diff_nodes;
  AliasDb db(graph);
  SubgraphSlicer(graph->block(), graph, threshold, db).run();
  SubgraphPostProcessor(threshold, diff_nodes).run(graph->block());

  // Run CSE to eliminate duplicates that may have occurred
  // while inlining subgraphs.
  EliminateCommonSubexpression(graph);
  return diff_nodes;
}
} // namespace jit
} // namespace torch
