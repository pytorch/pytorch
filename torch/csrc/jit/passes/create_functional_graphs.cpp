#include <torch/csrc/jit/passes/create_functional_graphs.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/utils/memory.h>

#include "torch/csrc/jit/ir.h"

#include <cstddef>
#include <limits>

namespace torch {
namespace jit {

namespace {

struct FunctionalGraphSlicer {
  FunctionalGraphSlicer(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  void run() {
    bool changed = true;
    // TODO: more sane strategy
    size_t MAX_NUM_ITERATIONS = 4;

    // First, analyze the functional subset of the graph, and then create
    // functional graphs. The graph gets mutated when we create functional
    // subgraphs, invalidating the AliasDb, so we need to do our analysis
    // first.
    for (size_t i = 0; i < MAX_NUM_ITERATIONS && changed; ++i) {
      aliasDb_ = torch::make_unique<AliasDb>(graph_);
      AnalyzeFunctionalSubset(graph_->block());
      changed = CreateFunctionalGraphsImpl(graph_->block());
    }
  }

 private:
  bool isEmptyFunctionalGraph(Node* n) {
    auto g = n->g(attr::Subgraph);
    return g->inputs().size() == 0 && g->outputs().size() == 0;
  }

  void nonConstNodes(Block* block, size_t* num) {
    for (auto it = block->nodes().begin();
         it != block->nodes().end() && *num < minSubgraphSize_;
         ++it) {
      Node* n = *it;
      if (n->kind() == prim::Constant) {
        continue;
      }
      *num = *num + 1;
      for (Block* b : n->blocks()) {
        nonConstNodes(b, num);
      }
    }
  }

  bool inlineIfTooSmall(Node* n) {
    AT_ASSERT(n->kind() == prim::FunctionalGraph);
    auto subgraph = SubgraphUtils::getSubgraph(n);
    size_t num_modes = 0;
    nonConstNodes(subgraph->block(), &num_modes);
    if (num_modes < minSubgraphSize_) {
      SubgraphUtils::unmergeSubgraph(n);
      return true;
    }
    return false;
  }

  bool CreateFunctionalGraphsImpl(Block* block) {
    /*
    Iterate the block in reverse and create FunctionalSubgraphs.
    When we encounter a node that isn't functional, we skip it. Otherwise,
    we try to merge the functional node into the current functional subgraph.
    If it can't be merged into the current functional subgraph node, then we
    start a functional subgraph group.
    */
    bool changed = false;
    std::vector<Node*> functional_nodes;

    Node* functional_subgraph_node =
        graph_->createWithSubgraph(prim::FunctionalGraph)
            ->insertBefore(block->return_node());
    bool seen_escape_value = false;
    auto reverse_iter = block->nodes().reverse();
    std::vector<Value*> graph_outputs;
    for (auto it = reverse_iter.begin(); it != reverse_iter.end();) {
      Node* n = *it++;

      // constants get copied into the graph
      if (n->kind() == prim::Constant || n == functional_subgraph_node) {
        continue;
      }

      // if `n` is functional, all of its blocks will be merged into the
      // new functional subgraph, so we only need to recurse if it is not
      // functional
      if (!functional_nodes_.count(n)) {
        for (Block* b : n->blocks()) {
          auto block_changed = CreateFunctionalGraphsImpl(b);
          changed = block_changed && changed;
        }
        continue;
      }

      auto new_escape_values = filter(n->outputs(), [&](Value* output) {
        return escape_values_.count(output);
      });
      TORCH_INTERNAL_ASSERT(new_escape_values.size() <= 1);
      bool new_escape_value = new_escape_values.size() == 1;

      if (n->kind() == prim::FunctionalGraph &&
          isEmptyFunctionalGraph(functional_subgraph_node)) {
        functional_subgraph_node->destroy();
        functional_subgraph_node = n;
        seen_escape_value = new_escape_value;
        continue;
      }

      changed = true;
      bool at_most_one_escape_value = !seen_escape_value || !new_escape_value;
      if (aliasDb_->moveBeforeTopologicallyValid(n, functional_subgraph_node) &&
          at_most_one_escape_value) {
        SubgraphUtils::mergeNodeIntoSubgraph(n, functional_subgraph_node);
        seen_escape_value = new_escape_value || seen_escape_value;
      } else {
        functional_nodes.emplace_back(functional_subgraph_node);
        seen_escape_value = new_escape_value;
        functional_subgraph_node =
            graph_->createWithSubgraph(prim::FunctionalGraph)->insertAfter(n);
        SubgraphUtils::mergeNodeIntoSubgraph(n, functional_subgraph_node);
      }
    }
    functional_nodes.emplace_back(functional_subgraph_node);

    for (Node* functional_node : functional_nodes) {
      if (!inlineIfTooSmall(functional_node)) {
        ConstantPooling(functional_node->g(attr::Subgraph));
      }
    }
    return changed;
  }

  bool AnalyzeFunctionalSubset(Node* n) {
    bool functional_node = true;
    for (Value* v : n->outputs()) {
      if (!aliasDb_->hasWriters(v)) {
        functional_values_.insert(v);
      } else {
        functional_node = false;
      }
    }
    for (Block* block : n->blocks()) {
      auto functional_block = AnalyzeFunctionalSubset(block);
      functional_node = functional_node && functional_block;
    }

    auto inputs = n->inputs();
    functional_node = functional_node &&
        std::all_of(inputs.begin(), inputs.end(), [&](Value* v) {
                        return !aliasDb_->hasWriters(v);
                      });

    // [ESCAPE VALUES]
    // Functional Graphs are not responsible for maintaining aliasing
    // relationships. If two outputs of a node escape the local scope, then we
    // might change semantics of the program if their aliasing relationships
    // is changed.
    // TODO: outputs which alias the wildcard set but do not "re-escape" can
    // also be considered functional.

    size_t escape_outputs = 0;
    for (Value* output : n->outputs()) {
      if (aliasDb_->escapesScope(output)) {
        escape_values_.insert(output);
        escape_outputs += 1;
      }
    }

    bool at_most_one_escape_outputs = escape_outputs <= 1;
    if (functional_node && at_most_one_escape_outputs) {
      functional_nodes_.insert(n);
      return true;
    } else {
      return false;
    }
  }

  void AnalyzeFunctionalSubset(at::ArrayRef<Block*> blocks) {
    for (Block* block : blocks) {
      AnalyzeFunctionalSubset(block);
    }
  }

  bool AnalyzeFunctionalSubset(Block* block) {
    bool is_functional_block = true;
    // block inputs will not yet have been iterated through
    for (Value* v : block->inputs()) {
      if (!aliasDb_->hasWriters(v) && !aliasDb_->escapesScope({v})) {
        functional_values_.insert(v);
      } else {
        is_functional_block = false;
      }
    }
    for (Node* n : block->nodes()) {
      bool functional = AnalyzeFunctionalSubset(n);
      is_functional_block = is_functional_block && functional;
    }
    is_functional_block = is_functional_block &&
        std::all_of(block->outputs().begin(),
                    block->outputs().end(),
                    [&](Value* v) { return functional_values_.count(v); });
    return is_functional_block;
  }

  std::unordered_set<Node*> functional_nodes_;
  std::unordered_set<Value*> functional_values_;
  std::unordered_set<Value*> escape_values_;
  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  size_t minSubgraphSize_ = 6;
};

void InlineFunctionalGraphs(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++;
    for (Block* b : n->blocks()) {
      InlineFunctionalGraphs(b);
    }
    if (n->kind() == prim::FunctionalGraph) {
      SubgraphUtils::unmergeSubgraph(n);
    }
  }
}

} // namespace

void CreateFunctionalGraphs(const std::shared_ptr<Graph>& graph) {
  // Run Constant Pooling so constants get hoisted
  ConstantPooling(graph);
  FunctionalGraphSlicer func(graph);
  func.run();
  // Creation of Functional Subgraphs & Deinlining creates excess constants
  ConstantPooling(graph);
}

void InlineFunctionalGraphs(const std::shared_ptr<Graph>& graph) {
  InlineFunctionalGraphs(graph->block());
}

} // namespace jit
} // namespace torch
