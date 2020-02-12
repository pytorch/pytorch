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
    bool functional_node = true;

    // Functional Graphs are not responsible for maintaining aliasing
    // relationships. If an output of a functional graph escapes scope
    // or is mutated then we might change semantics of the program if
    // aliasing relationships are changed.
    // For now, we don't allow any values which are mutated into the functional
    // graph, and we don't allow any nodes which have outputs that escape scope.
    // Possible Future Improvements:
    // - allow inputs to have mutations so long as there are no mutations in the
    // graph
    // - allow functional graphs to have at most one value that can escape scope
    // - allow outputs which alias the wildcard set but do not "re-escape"

    for (Value* v : n->outputs()) {
      bool has_writers = aliasDb_->hasWriters(v);
      bool escapes_scope = aliasDb_->escapesScope(v);
      if (has_writers) {
        mutated_values_.insert(v);
      }
      functional_node = functional_node && !escapes_scope && !has_writers;
    }

    for (Block* block : n->blocks()) {
      auto functional_block = AnalyzeFunctionalSubset(block);
      functional_node = functional_node && functional_block;
    }

    auto inputs = n->inputs();
    functional_node = functional_node &&
        std::all_of(inputs.begin(), inputs.end(), [&](Value* v) {
                        return !mutated_values_.count(v);
                      });
    if (functional_node) {
      functional_nodes_.insert(n);
    }
    return functional_node;
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

struct MutationRemover {
  MutationRemover(const std::shared_ptr<Graph>& graph)
      : aliasDb_(nullptr), graph_(graph) {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
  }

  void run() {
    RemoveAtenMutation(graph_->block());
    RemoveListMutation(graph_->block());
  }

 private:
  bool uniqueAlias(Value* v) {
    // if the output isn't contained or alias by the inputs to its node, it's
    // unique
    bool unique_alias = !aliasDb_->mayContainAlias(v->node()->inputs(), v);
    // bail on nodes with side effects like prim::If etc
    return unique_alias && !v->node()->hasSideEffects();
  }

  bool inplaceOpVariant(Node* n) {
    if (!n->kind().is_aten()) {
      return false;
    }
    auto name = n->schema().name();
    bool inplace_op = name.at(name.size() - 1) == '_';
    if (!inplace_op) {
      return false;
    }
    auto new_schema = name.substr(0, name.size() - 1);
    return getAllOperatorsFor(Symbol::fromQualString(new_schema)).size() != 0;
  }

  bool listAppendFollowingListConstruct(Node* n) {
    return n->kind() == aten::append &&
        n->inputs().at(0)->node()->kind() == prim::ListConstruct;
  }

  bool tryMakeCreationAndMutationAtomic(
      Value* mutated_value,
      Node* mutating_op) {
    // We can only remove mutation to values that are unique aliases in the
    // graph. if x = y[0] or y = self.y, then removing the mutation could
    // change observable semantics
    if (!uniqueAlias(mutated_value)) {
      return false;
    }

    // In order to safely remove a mutation, the creation of a tensor and its
    // subsequent mutation need to be one atomic operation
    return aliasDb_->moveBeforeTopologicallyValid(
        mutated_value->node(), mutating_op);
  }

  void RemoveListMutation(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      auto* node = *it;
      it++;

      for (Block* sub_block : node->blocks()) {
        RemoveListMutation(sub_block);
      }

      if (!listAppendFollowingListConstruct(node)) {
        continue;
      }

      Value* mutated_value = node->inputs().at(0);
      if (!tryMakeCreationAndMutationAtomic(mutated_value, node)) {
        continue;
      }

      Node* list_construct = mutated_value->node();
      list_construct->addInput(node->inputs().at(1));
      node->output()->replaceAllUsesWith(mutated_value);
      node->destroy();

      // :(  TODO: incremental update ?
      aliasDb_ = torch::make_unique<AliasDb>(graph_);
    }
  }

  void RemoveAtenMutation(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      auto* node = *it;
      it++;

      for (Block* sub_block : node->blocks()) {
        RemoveAtenMutation(sub_block);
      }

      // TODO: out op variants
      if (!inplaceOpVariant(node)) {
        continue;
      }

      Value* mutated_value = node->inputs().at(0);
      if (!tryMakeCreationAndMutationAtomic(mutated_value, node)) {
        continue;
      }

      // all inplace ops at time of writing have a single output. new ops with
      // multiple outputs could have strange aliasing cases so only handle the
      // single output case
      if (node->outputs().size() != 1) {
        continue;
      }

      auto schema_name = node->schema().name();
      auto new_schema = schema_name.substr(0, schema_name.size() - 1);
      auto new_node = graph_->create(Symbol::fromQualString(new_schema), 1);
      new_node->insertBefore(node);
      for (Value* input : node->inputs()) {
        new_node->addInput(input);
      }
      new_node->output()->setType(node->output()->type());
      mutated_value->replaceAllUsesAfterNodeWith(node, new_node->output());
      node->output()->replaceAllUsesWith(new_node->output());
      node->destroy();

      // :(  TODO: incremental update ?
      aliasDb_ = torch::make_unique<AliasDb>(graph_);
    }
  }

 private:
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::shared_ptr<Graph> graph_;
};

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

void RemoveMutation(const std::shared_ptr<Graph>& graph) {
  MutationRemover mr(graph);
  mr.run();
}

} // namespace jit
} // namespace torch
