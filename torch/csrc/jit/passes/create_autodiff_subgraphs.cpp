#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"

#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"

namespace torch {
namespace jit {

namespace {

class SubgraphSlicer {
 public:
  SubgraphSlicer(Block* block, size_t minSubgraphSize)
      : block_(block), minSubgraphSize_(minSubgraphSize) {}

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
      for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();) {
        bool changed;
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }

    auto curNode = *block_->nodes().rbegin();
    while (curNode != *block_->nodes().rend()) {
      for (auto subBlock : curNode->blocks()) {
        SubgraphSlicer(subBlock, minSubgraphSize_).run(diffGraphs);
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
    // Run CSE one more time to eliminate duplicates that may have occured
    // while re-inlining subgraphs.
    EliminateCommonSubexpression(block_);
  }

 private:
  static Graph& getSubgraph(Node* n) {
    JIT_ASSERT(n->kind() == prim::DifferentiableGraph);
    return *n->g(attr::Subgraph);
  }

  // Inline this node's group subgraph into the outer graph if it's smaller
  // than the specified minimum size.
  //
  // Returns true if an inlining has occured, false otherwise.
  bool inlineIfTooSmall(Node* n) {
    JIT_ASSERT(n->kind() == prim::DifferentiableGraph);
    auto& subgraph = getSubgraph(n);
    size_t i = 0;
    for (auto it = subgraph.nodes().begin(); it != subgraph.nodes().end();
         ++it) {
      if (++i >= minSubgraphSize_) {
        return false;
      }
    }

    unmergeGroup(n);
    return true;
  }

  // Move nodes from producerGroup's subgraph to the top-level graph.
  // This destroys `producerGroup`.
  std::vector<Node*> unmergeGroup(Node* producerGroup) {
    JIT_ASSERT(producerGroup->kind() == prim::DifferentiableGraph);

    std::vector<Node*> temporary_nodes;
    auto& producerSubgraph = getSubgraph(producerGroup);

    // Initialize a map of inner graph values to outer graph values
    std::unordered_map<const Value*, Value*> innerToOuter;
    const auto innerInputs = producerSubgraph.inputs();
    const auto outerInputs = producerGroup->inputs();
    for (size_t i = 0; i < innerInputs.size(); ++i) {
      innerToOuter[innerInputs[i]] = outerInputs[i];
    }

    // Clone all nodes
    for (auto inner : producerSubgraph.nodes()) {
      Node* outer = block_->owningGraph()->createClone(
          inner, [&](Value* k) -> Value* { return innerToOuter.at(k); });
      outer->insertBefore(producerGroup);
      temporary_nodes.emplace_back(outer);
      const auto innerOutputs = inner->outputs();
      const auto outerOutputs = outer->outputs();
      for (size_t i = 0; i < innerOutputs.size(); ++i)
        innerToOuter[innerOutputs[i]] = outerOutputs[i];
    }

    // Replace uses of producerGroup outputs and destroy the producer
    const auto subgraphOutputs = producerSubgraph.outputs();
    for (size_t i = 0; i < subgraphOutputs.size(); ++i) {
      const auto outerOutput = innerToOuter.at(subgraphOutputs[i]);
      producerGroup->outputs()[i]->replaceAllUsesWith(outerOutput);
    }
    producerGroup->destroy();

    return temporary_nodes;
  }

  // Combine the nodes in two groups together. The nodes will end up in
  // `consumerGroup`, and `producerGroup` will be deleted.
  void mergeGroups(Node* consumerGroup, Node* producerGroup) {
    // Extract the nodes in `producerGroup` into the outer graph
    const auto nodes = unmergeGroup(producerGroup);
    // Then merge them into `consumerGroup`
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
      mergeNodeIntoGroup(consumerGroup, *it);
    }
  }

  // Merge node `n` into `group`'s subgraph
  Node* mergeNodeIntoGroup(Node* group, Node* n) {
    JIT_ASSERT(group->kind() == prim::DifferentiableGraph);
    JIT_ASSERT(n->kind() != prim::DifferentiableGraph);

    auto& subgraph = getSubgraph(group);

    // Map from values in the surrounding graph to inputs in the subgraph
    std::unordered_map<Value*, Value*> inputsMap;

    JIT_ASSERT(group->inputs().size() == subgraph.inputs().size());
    size_t i = 0;
    for (auto input : group->inputs()) {
      inputsMap[input] = subgraph.inputs()[i];
      i++;
    }

    // Add n's inputs to the group's input list if we don't already have them
    WithInsertPoint guard(*subgraph.nodes().begin());
    for (auto input : n->inputs()) {
      if (inputsMap.count(input) == 0) {
        // Clone constants inside the subgraph instead of referencing them, to
        // enable more optimizations
        if (auto value = toIValue(input)) {
          auto nv = subgraph.insertConstant(*value);
          inputsMap[input] = nv;
        } else {
          // The common case: this is a regular input, so just register it with
          // the group node and inner subgraph
          group->addInput(input);
          auto inputToGraph = subgraph.addInput();
          inputToGraph->setType(input->type());
          inputsMap[input] = inputToGraph;
        }
      }
    }

    // Merge the node into the graph
    auto mergedNode = subgraph.insertNode(
        subgraph.createClone(n, [&](Value* v) { return inputsMap[v]; }));

    // If n's outputs were inputs to `group`, remove them since we just merged
    // n in.
    //
    // i.e.,
    // x = f(w); group(x, y, z) becomes group(w, y, z).
    // x, y, z = f(w); group(x, y, z) becomes group(w).
    auto inputs = group->inputs();
    for (size_t i = 0; i < n->outputs().size(); ++i) {
      auto it = std::find(inputs.begin(), inputs.end(), n->outputs()[i]);
      if (it != inputs.end()) {
        size_t p = it - inputs.begin();
        group->removeInput(p);
        subgraph.inputs()[p]->replaceAllUsesWith(mergedNode->outputs()[i]);
        subgraph.eraseInput(p);
      }
    }

    // Add n's outputs to the group node and inner subgraph outputs.
    for (size_t i = 0; i < n->outputs().size(); i++) {
      auto oldOutput = n->outputs()[i];

      // Only register the output in the group node if it's actually used
      // outside the subgraph.
      const auto hasUsesOutsideGroup = std::any_of(
          oldOutput->uses().cbegin(),
          oldOutput->uses().cend(),
          [&](const Use& use) { return use.user->isAfter(group); });

      if (hasUsesOutsideGroup) {
        auto newOutput = mergedNode->outputs()[i];
        subgraph.registerOutput(newOutput);
        auto groupOutput = group->addOutput();
        groupOutput->copyMetadata(oldOutput);
        oldOutput->replaceAllUsesWith(groupOutput);
      }
    }

    // Remove the original node now that the merge is complete
    n->destroy();

    return mergedNode;
  }

  // Create a group node that contains only `n`
  // `n` is destroyed.
  Node* createSingletonGroup(Node* n) {
    auto group = block_->owningGraph()->createDifferentiableSubgraph();
    group->insertBefore(n);
    mergeNodeIntoGroup(group, n);
    return group;
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
        consumer = createSingletonGroup(consumer);
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
    JIT_ASSERT(consumer->kind() == prim::DifferentiableGraph);
    bool canMerge = shouldConsiderForMerge(producer) &&
        producer->moveBeforeTopologicallyValid(consumer);

    if (!canMerge) {
      return c10::nullopt;
    }

    if (producer->kind() == prim::DifferentiableGraph) {
      mergeGroups(consumer, producer);
    } else {
      mergeNodeIntoGroup(consumer, producer);
    }

    return consumer;
  }

  Block* block_;
  size_t minSubgraphSize_;
};
} // anonymous namespace

std::vector<Node*> CreateAutodiffSubgraphs(Graph& graph, size_t threshold) {
  std::vector<Node*> diff_nodes;
  SubgraphSlicer(graph.block(), threshold).run(diff_nodes);
  return diff_nodes;
}

} // namespace jit
} // namespace torch
