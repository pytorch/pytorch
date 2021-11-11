#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/remove_redundant_profiles.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include "ATen/core/interned_strings.h"
namespace torch {
namespace jit {

namespace {

static c10::Symbol dont_merge_sym = Symbol::attr("dont_merge");

struct WorkBlock : public std::pair<Node*, Node*> {
  using pair::pair;

  Node* begin() {
    return this->first;
  }
  Node* end() {
    return this->second;
  }
};

struct topo_cmp_value {
  bool operator()(Value* a, Value* b) const {
    return a->node()->isBefore(b->node());
  }
};

struct topo_cmp_node {
  bool operator()(Node* a, Node* b) const {
    return a->isBefore(b);
  }
};


class SubgraphSlicer {
 public:
  SubgraphSlicer(
      Block* block,
      std::shared_ptr<Graph> graph,
      size_t minSubgraphSize,
      AliasDb& aliasDb,
      std::vector<Node*>& diff_nodes)
      : block_(block),
        graph_(std::move(graph)),
        minSubgraphSize_(minSubgraphSize),
        aliasDb_(aliasDb),
        diff_nodes_(diff_nodes) {}

  void run() {
    // We maintain alias db correctness in-place while building up the autodiff
    // subgraphs, however it is difficult to preserve correctness when
    // un-inlining autodiff subgraphs. We first recursively construct all
    // subgraphs and then recursively cleanup & unmerge the small subgraphs
    buildupSubgraphs();
    cleanupSubgraphs();
    // Run CSE globally onceto eliminate duplicates that may have occurred
    // while inlining subgraphs.
    EliminateCommonSubexpression(graph_);
  }

  void cleanupSubgraphs() {
    auto curNode = *block_->nodes().rbegin();

    bool any_changed = true;

    while(any_changed) {
        any_changed = false;
        while (curNode != *block_->nodes().rend()) {
          // Save the previous node, since we might delete `curNode` in next block
          auto prevNode = curNode->prev();
          if (curNode->kind() == prim::DifferentiableGraph) {

            // aliased outputs in DifferentiableGraphs must be unfused
            // since autodiff doesn't know how to handle them correctly
            any_changed = unfuseAliasedOutputs(curNode);
            // Inlining nodes may cause some subexpression to come back in the
            // subgraphs (for example, copying constants in repeatedly will generate
            // redundant prim::Constants). Run CSE to clean them up.
            EliminateCommonSubexpression(curNode->g(attr::Subgraph));
            // don't try inlining until we unfuse all unaliased outputs
            if (!any_changed && !inlineIfTooSmall(curNode)) {
              diff_nodes_.push_back(curNode);
            }
          }
          curNode = prevNode;
      }

    }

    for (Node* n : block_->nodes()) {
      for (Block* b : n->blocks()) {
        SubgraphSlicer(b, graph_, minSubgraphSize_, aliasDb_, diff_nodes_)
            .cleanupSubgraphs();
      }
    }
  }

  void buildupSubgraphs() {
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
          // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
          bool changed;
          std::tie(it, changed) = scanNode(*it);
          any_changed |= changed;
        }
      }
    }

    // Construct Subgraphs Recursively
    for (Node* n : block_->nodes()) {
      for (auto subBlock : n->blocks()) {
        SubgraphSlicer(
            subBlock, graph_, minSubgraphSize_, aliasDb_, diff_nodes_)
            .buildupSubgraphs();
      }
    }
  }

 private:

  bool unfuseAliasedOutputs(Node* subgraphNode) {

    GRAPH_DEBUG("unfuseAliasedOutputs on ", getHeader(subgraphNode));
    if (subgraphNode->outputs().size() < 2) {
      return false;
    }

    auto subgraph = subgraphNode->g(attr::Subgraph);
    GRAPH_DUMP("unfuseAliasedOutputs Subgraph ", subgraph);
    auto sets = buildAliasedSets(subgraph);
    GRAPH_DEBUG("buildAliasedSets sets.size() = ", sets.size());

    std::set<Node*, topo_cmp_node> nodes;

    for (auto i : c10::irange(sets.size())) {
      if (sets[i].size() <= 1) {
        continue;
      }

      // we have at least two aliased outputs
      // we skip the earliest node w.r.t. the topo order
      // NB. after some nodes are unfused, the outputs of some other nodes
      // may become the outputs of the subgraph and alias the remaining ones
      // so we have to re-run this function until there are no more changes
      auto it = ++sets[i].begin();
      while (it != sets[i].end()) {
        GRAPH_DEBUG("root aliased value ", (*it)->debugName(), " node ", *(*it)->node());
        collectNodesToUnfuse((*it)->node(), nodes);
        it++;
      }
    }

    // unfuse in the reverse topo order
    for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
      unfuseNode(*it, subgraphNode);
    }

    if (!nodes.empty()) {
      GRAPH_DUMP("after unfusing aliased nodes: ", graph_);
    }

    return !nodes.empty();
  }

  void collectNodesToUnfuse(Node* start, std::set<Node*, topo_cmp_node>& s) {
    if(start->kind() == prim::Return || start->kind() == prim::Param) {
      GRAPH_DEBUG("reached the param or return node", getHeader(start));
      return;
    }

    if (s.count(start) != 0) {
      // already visited, no need to visit descendants
      return;
    }

    GRAPH_DEBUG("collectNodesToUnfuse: inserting node ", getHeader(start));
    s.insert(start);

    for (auto o : start->outputs()) {
      for (auto use : o->uses()) {
        collectNodesToUnfuse(use.user, s);
      }
    }
  }

  void unfuseNode(Node* n, Node* subgraphNode) {
    // collect output indices

    GRAPH_DEBUG("unfuseNode node ", getHeader(n));
    auto subgraph = n->owningGraph();
    //auto subgraphNode = n->owningBlock()->owningNode();

    std::set<Value*> node_outputs(n->outputs().begin(), n->outputs().end()); 
    std::set<size_t> output_indices;
    std::set<Value*> node_inputs(n->inputs().begin(), n->inputs().end());


    std::unordered_map<Value*, Value*> local_map;
    auto env = [&](Value* v) {
      auto it = local_map.find(v);
      if (it != local_map.end()) {
        return it->second;
      }
      TORCH_INTERNAL_ASSERT(false, "all inputs should've been mapped. Couldn't map %", v->debugName());
      return v;
    };

    for (auto i : c10::irange(subgraph->outputs().size())) {
      if (node_outputs.count(subgraph->outputs().at(i)) != 0) {
        output_indices.insert(i);
      }
      
      if (node_inputs.count(subgraph->outputs().at(i)) != 0) {
        GRAPH_DEBUG("output %", subgraph->outputs().at(i)->debugName(), " is already subgraph's output");
        GRAPH_DEBUG("Mapping %", subgraph->outputs().at(i), " to %", subgraphNode->outputs().at(i));
        local_map[subgraph->outputs().at(i)] = subgraphNode->outputs().at(i);
        node_inputs.erase(subgraph->outputs().at(i));
      } 
    }

    WithInsertPoint wip(subgraphNode->next());

    // these node inputs need to be added to subgraph's outputs
    // put them in vmap
    for (auto ni : node_inputs) {

      if (local_map.count(ni) != 0) {
        // this could happen if `n` uses two or more outputs
        // of a constant node and we already clone the constant
        // into the outer graph and mapped its outputs
        continue;
      }

      Value* sno = nullptr;
      if (ni->node()->kind() == prim::Constant) {
        auto copy = subgraphNode->owningGraph()->createClone(ni->node(), env);
        subgraphNode->owningGraph()->insertNode(copy);
        // in case we have a multi-output const, map the rest of the outputs
        // so we get to clone `n`, `n`'s clone will use the outputs of the 
        // constant clone
        for (auto i : c10::irange(n->outputs().size())) {
          GRAPH_DEBUG("Mapping %", ni->node()->output(i)->debugName(), " to %", copy->output(i)->debugName());
          local_map[ni->node()->output(i)] = copy->output(i);
        }
      }
      else {
        subgraph->registerOutput(ni);
        sno = subgraphNode->addOutput();
        sno->setType(ni->type());
        GRAPH_DEBUG("Mapping %", ni->debugName(), " to %", sno->debugName());
        local_map[ni] = sno;
      }
    }

    auto copy = subgraphNode->owningGraph()->createClone(n, env);
    GRAPH_DEBUG("copy ", *copy);

    for (auto i : c10::irange(n->outputs().size())) {
      auto oo = n->outputs()[i];
      auto no = copy->outputs()[i];
      no->copyMetadata(oo);
      GRAPH_DEBUG("Mapping %", oo->debugName(), " to %", no->debugName());
      local_map[oo] = no;    
    }

    subgraphNode->owningGraph()->insertNode(copy);

    for (auto it = output_indices.rbegin(); it != output_indices.rend(); it++) {
      auto replace_val = local_map[subgraph->outputs().at(*it)];
      subgraphNode->outputs().at(*it)->replaceAllUsesWith(replace_val);
      subgraphNode->eraseOutput(*it);
      subgraph->eraseOutput(*it);
    }

    // size_t num_outputs = subgraph->outputs().size();
    // for (int64_t i = num_outputs - 1; i >= 0; i++) {
    //   subgraphNode->eraseOutput(i);
    //   subgraph->eraseOutput(i);
    // }

    n->destroy();
  }

  std::vector<std::set<Value*, topo_cmp_value>> buildAliasedSets(std::shared_ptr<Graph> subgraph) {
    auto outputs = subgraph->outputs();
    AliasDb alias_db(subgraph);
    TORCH_INTERNAL_ASSERT(outputs.size() > 1);
    std::vector<std::set<Value*, topo_cmp_value>> res;
    for (auto o : outputs) {
      auto grouped = false;
      for (auto& s : res) {
        auto os = *s.begin();
        GRAPH_DEBUG("comparing %", o->debugName(), " with %", os->debugName(), " result ", (alias_db.mayContainAlias(os, o) || alias_db.mayContainAlias(o, os)));
        if (alias_db.mayContainAlias(os, o) ||
            alias_db.mayContainAlias(o, os) || 
            alias_db.mayAlias(os, o) ||
            alias_db.mayAlias(o, os)) {
          s.insert(o);
          GRAPH_DEBUG("Grouping %", o->debugName(), " with %", os->debugName());
          grouped = true;
        }
      }
      if (!grouped) {
        res.push_back({o});
      }
    }
    return res;
  }

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
      i += !it->notExecutedOp();
      if (i >= minSubgraphSize_) {
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

    if (node->hasAttribute(dont_merge_sym)) {
      return false;
    }

    return isDifferentiable(node);
  }

  std::pair<graph_node_list::iterator, bool> scanNode(Node* consumer) {
    if (shouldConsiderForMerge(consumer)) {
      if (consumer->kind() != prim::DifferentiableGraph) {
        consumer = SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
            consumer, prim::DifferentiableGraph, aliasDb_);
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

    SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
        producer, consumer, aliasDb_);
    return consumer;
  }

  Block* block_;
  std::shared_ptr<Graph> graph_;
  size_t minSubgraphSize_;
  AliasDb& aliasDb_;
  std::vector<Node*>& diff_nodes_;
};
} // anonymous namespace

std::vector<Node*> CreateAutodiffSubgraphs(
    const std::shared_ptr<Graph>& graph,
    size_t threshold) {
  std::vector<Node*> diff_nodes;
  AliasDb db(graph);
  GRAPH_DEBUG("Before removing redundant profiles", *graph);
  RemoveRedundantProfiles(graph->block(), db);
  GRAPH_DEBUG("Before creating autodiff subgraphs", *graph);
  SubgraphSlicer(graph->block(), graph, threshold, db, diff_nodes).run();
  GRAPH_DEBUG("After creating autodiff subgraphs", *graph);
  return diff_nodes;
}
} // namespace jit
} // namespace torch
