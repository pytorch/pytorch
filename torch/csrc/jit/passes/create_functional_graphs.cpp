#include <torch/csrc/jit/passes/create_functional_graphs.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

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
      aliasDb_ = std::make_unique<AliasDb>(graph_);
      AnalyzeFunctionalSubset(graph_->block());
      changed = CreateFunctionalGraphsImpl(graph_->block());
    }
  }

 private:
  bool isEmptyFunctionalGraph(Node* n) {
    auto g = n->g(attr::Subgraph);
    return g->inputs().empty() && g->outputs().empty();
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
    std::vector<Node*> functional_graph_nodes;

    Node* functional_subgraph_node =
        graph_->createWithSubgraph(prim::FunctionalGraph)
            ->insertBefore(block->return_node());
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

      if (n->kind() == prim::FunctionalGraph &&
          isEmptyFunctionalGraph(functional_subgraph_node)) {
        functional_subgraph_node->destroy();
        functional_subgraph_node = n;
        continue;
      }

      changed = true;
      if (aliasDb_->moveBeforeTopologicallyValid(n, functional_subgraph_node)) {
        SubgraphUtils::mergeNodeIntoSubgraph(n, functional_subgraph_node);
      } else {
        functional_graph_nodes.emplace_back(functional_subgraph_node);
        functional_subgraph_node =
            graph_->createWithSubgraph(prim::FunctionalGraph)->insertAfter(n);
        SubgraphUtils::mergeNodeIntoSubgraph(n, functional_subgraph_node);
      }
    }
    functional_graph_nodes.emplace_back(functional_subgraph_node);

    for (Node* functional_node : functional_graph_nodes) {
      if (!inlineIfTooSmall(functional_node)) {
        ConstantPooling(functional_node->g(attr::Subgraph));
      }
    }
    return changed;
  }

  bool AnalyzeFunctionalSubset(Node* n) {
    // TODO: clarify hasSideEffects, isNondeterministic
    bool is_functional_node = true;

    // Functional Graphs are not responsible for maintaining aliasing
    // relationships. If an output of a functional graph escapes scope
    // or is mutated then we might change semantics of the program if
    // aliasing relationships are changed.
    // We don't allow any node in the functional graph to output a value
    // that escapes scope or is mutated, and we don't allow any mutating nodes
    // into the graph.
    // - allow functional graphs to have at most one value that can escape scope
    // - allow outputs which alias the wildcard set but do not "re-escape"
    for (Value* v : n->outputs()) {
      bool has_writers = aliasDb_->hasWriters(v);
      bool escapes_scope = aliasDb_->escapesScope(v);
      if (has_writers) {
        mutated_values_.insert(v);
      }
      is_functional_node = is_functional_node && !escapes_scope && !has_writers;
    }

    for (Block* block : n->blocks()) {
      auto functional_block = AnalyzeFunctionalSubset(block);
      is_functional_node = is_functional_node && functional_block;
    }

    is_functional_node = is_functional_node && !aliasDb_->isMutable(n);
    if (is_functional_node) {
      functional_nodes_.insert(n);
    }
    return is_functional_node;
  }

  void AnalyzeFunctionalSubset(at::ArrayRef<Block*> blocks) {
    for (Block* block : blocks) {
      AnalyzeFunctionalSubset(block);
    }
  }

  bool AnalyzeFunctionalSubset(Block* block) {
    bool is_functional_block = true;
    // block inputs will not yet have been iterated through,
    // so we need to add them to our set of mutated & escape values.
    for (Value* v : block->inputs()) {
      bool has_writers = aliasDb_->hasWriters(v);
      if (has_writers) {
        mutated_values_.insert(v);
      }
    }
    // if a block output is not functional, then the corresponding output for
    // the node that contains the block will not be functional either, so we do
    // not need to analyze the block outputs here.
    for (Node* n : block->nodes()) {
      bool functional = AnalyzeFunctionalSubset(n);
      is_functional_block = is_functional_block && functional;
    }
    return is_functional_block;
  }

  std::unordered_set<Node*> functional_nodes_;
  std::unordered_set<Value*> mutated_values_;
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
