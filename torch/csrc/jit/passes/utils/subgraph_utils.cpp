#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

#include <torch/csrc/jit/passes/canonicalize.h>

#include <ATen/core/symbol.h>
#include <c10/util/StringUtil.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>

#include <utility>

namespace torch {
namespace jit {
namespace SubgraphUtils {
namespace {

bool hasSubgraph(Node* n) {
  return n->hasAttribute(attr::Subgraph);
}

std::vector<c10::optional<const Use>> gatherLastUses(
    at::ArrayRef<Value*> values) {
  return fmap(values, [&](Value* v) -> c10::optional<const Use> {
    return firstOrLastUse(v, /*find_first*/ false);
  });
}

// When merging a node into a subgraph, we wish to preserve all of the
// aliasing properties of the node's outputs. It is difficult to track
// the node or its contained nodes through all of the ir manipulation
// involved in merging; it is pretty easy to uniquely identify the value
// based on its uses. We can identify the value by its last use in the graph.
// Values which do not have uses or which do not have a last use
// outside of the subgraph to be merged into we do not need to track.
struct ValueMapper {
  // `to_merge` is the node we're merginginto a subgraph, `existing_subgraph` is
  // the subgraph node that we're merging into if it exists
  ValueMapper(
      Node* to_merge,
      AliasDb& db,
      c10::optional<Node*> existing_subgraph) {
    last_uses_ = gatherLastUses(to_merge->outputs());
    if (existing_subgraph) {
      existing_last_uses_ = gatherLastUses((*existing_subgraph)->outputs());
    }
    WithInsertPoint guard(to_merge);
    auto g = to_merge->owningGraph();
    // temporary node to put the aliasing properties of the node before its
    // merged and destroyed
    placeholder_node_ = g->insertNode(g->create(prim::Uninitialized, 0));
    for (size_t i = 0; i < to_merge->outputs().size(); ++i) {
      Value* existing = to_merge->outputs().at(i);
      Value* new_value = placeholder_node_->insertOutput(i)->copyMetadata(
          to_merge->outputs().at(i));
      db.replaceWithNewValue(existing, new_value);
    }
  }

  bool usesEqual(const Use& a, const Use& b) {
    return a.user == b.user && a.offset == b.offset;
  }

  void copyAliasing(Node* merged_node, AliasDb& db) {
    auto new_outputs = merged_node->outputs();
    for (Value* v : new_outputs) {
      auto maybe_last_use = firstOrLastUse(v, /*find_first*/ false);
      // if it doesnt have a use it shouldnt have been added as output
      TORCH_INTERNAL_ASSERT(maybe_last_use);
      const Use last_use = *maybe_last_use;

      // existing outputs of the subgraph do not need to have alias db mappings
      // updated
      bool is_existing_value = false;
      for (size_t i = 0; i < existing_last_uses_.size() && !is_existing_value;
           ++i) {
        is_existing_value = existing_last_uses_[i].has_value() &&
            usesEqual(*existing_last_uses_[i], last_use);
      }
      if (is_existing_value) {
        continue;
      }

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
  std::vector<c10::optional<const Use>> existing_last_uses_;
  Node* placeholder_node_;
};

Node* executeSubgraphMergeAndUpdateAliasing(
    Node* to_merge,
    c10::optional<Node*> existing,
    AliasDb& db,
    const std::function<Node*(void)>& merge_fn) {
  // When we merge a node into a subgraph, the new subgraph outputs
  // have the same aliasing properties as the original node's outputs.
  // Here we create a placeholder node, transfer the aliasing properties
  // to the placeholder, execute the merge, and transfer the aliasing
  // properties to the appropriate fusion group outputs
  ValueMapper vm(to_merge, db, existing);
  Node* fusion_group = merge_fn();
  vm.copyAliasing(fusion_group, db);
  return fusion_group;
}

// Combine the nodes in two subgraph together. The nodes will end up in
// `mergeTo`, and `mergeFrom` is destroyed.
void mergeSubgraph(Node* mergeTo, Node* mergeFrom) {
  bool merge_from_is_after = mergeFrom->isAfter(mergeTo);
  Node* nodeBeforeMergeFrom = mergeFrom->prev();
  Node* nodeAfterMergeFrom = mergeFrom->next();

  unmergeSubgraph(mergeFrom);

  graph_node_list_iterator end_it;
  graph_node_list_iterator it;

  if (merge_from_is_after) {
    it = nodeBeforeMergeFrom->iterator();
    end_it = nodeAfterMergeFrom->iterator();
  } else {
    end_it = nodeBeforeMergeFrom->reverseIterator();
    it = nodeAfterMergeFrom->reverseIterator();
  }
  ++it;

  std::vector<Node*> merged_nodes;
  while (it != end_it) {
    Node* node = *it;
    ++it;
    mergeNodeIntoSubgraph(node, mergeTo);
  }
}

struct topo_cmp_value {
  bool operator()(Value* a, Value* b) const {
    if (a->node() == b->node()) {
      return a->unique() < b->unique();
    }
    return a->node()->isBefore(b->node());
  }
};

struct topo_cmp_node {
  bool operator()(Node* a, Node* b) const {
    return a->isBefore(b);
  }
};

void collectNodesToUnfuse(Node* start, std::set<Node*, topo_cmp_node>& s) {
  if (start->kind() == prim::Return || start->kind() == prim::Param) {
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

std::vector<std::set<Value*, topo_cmp_value>> buildAliasedSets(
    std::shared_ptr<Graph> subgraph) {
  auto outputs = subgraph->outputs();
  AliasDb alias_db(std::move(subgraph));
  TORCH_INTERNAL_ASSERT(outputs.size() > 1);
  std::vector<std::set<Value*, topo_cmp_value>> res;
  for (auto o : outputs) {
    auto grouped = false;
    for (auto& s : res) {
      auto os = *s.begin();
      auto aliased = alias_db.mayContainAlias(os, o);
      GRAPH_DEBUG(
          "comparing %",
          o->debugName(),
          " with %",
          os->debugName(),
          " result ",
          aliased);
      if (aliased) {
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

} // namespace

std::shared_ptr<Graph> getSubgraph(Node* n) {
  return n->g(attr::Subgraph);
}

void unmergeSubgraph(Node* subgraphNode) {
  // Inline the graph, replace uses of node outputs and destroy the node
  auto outerGraph = subgraphNode->owningGraph();
  WithInsertPoint guard(subgraphNode);
  const auto subgraphOutputs = insertGraph(
      *outerGraph, *getSubgraph(subgraphNode), subgraphNode->inputs());
  AT_ASSERT(subgraphOutputs.size() >= subgraphNode->outputs().size());
  for (size_t i = 0; i < subgraphNode->outputs().size(); ++i) {
    subgraphNode->outputs()[i]->replaceAllUsesWith(subgraphOutputs[i]);
  }
  subgraphNode->destroy();
}

void collectNestedUses(
    std::unordered_set<Value*>& closed_over_values,
    std::unordered_set<Value*>& new_values,
    std::unordered_map<Value*, Value*>& externalValuesMap,
    Node* input_node) {
  for (auto input : input_node->inputs()) {
    if (externalValuesMap.count(input) == 0 && new_values.count(input) == 0) {
      closed_over_values.insert(input);
    }
  }
  if (input_node->kind() == prim::If) {
    for (Block* block : input_node->blocks()) {
      for (Node* node : block->nodes()) {
        collectNestedUses(
            closed_over_values, new_values, externalValuesMap, node);
      }
      for (Value* v : block->outputs()) {
        if (externalValuesMap.count(v) == 0 && new_values.count(v) == 0) {
          closed_over_values.insert(v);
        }
      }
    }
  } else if (input_node->kind() == prim::Loop) {
    for (Value* v : input_node->inputs()) {
      if (externalValuesMap.count(v) == 0 && new_values.count(v) == 0) {
        closed_over_values.insert(v);
      }
    }
    Block* block = input_node->blocks().at(0);
    for (Value* v : block->inputs()) {
      new_values.insert(v);
    }
    for (Node* node : block->nodes()) {
      collectNestedUses(
          closed_over_values, new_values, externalValuesMap, node);
    }
  } else if (!input_node->blocks().empty()) {
    TORCH_INTERNAL_ASSERT(false, input_node, " kind not handled yet");
  }
  for (Value* output : input_node->outputs()) {
    new_values.insert(output);
  }
}

std::unordered_set<Value*> closedOverValues(
    Node* toMerge,
    std::unordered_map<Value*, Value*>& externalValuesMap) {
  std::unordered_set<Value*> closed_over_values;
  std::unordered_set<Value*> new_values;
  collectNestedUses(closed_over_values, new_values, externalValuesMap, toMerge);
  return closed_over_values;
}

void mergeNodeIntoSubgraph(
    Node* toMerge,
    Node* subgraphNode,
    bool destroyNode) {
  AT_ASSERT(hasSubgraph(subgraphNode) && toMerge != subgraphNode);
  if (hasSubgraph(toMerge)) {
    return mergeSubgraph(subgraphNode, toMerge);
  }

  auto subgraph = getSubgraph(subgraphNode);

  // Map from values in the surrounding graph to inputs/outputs in the subgraph
  std::unordered_map<Value*, Value*> externalValuesMap;

  AT_ASSERT(subgraphNode->inputs().size() == subgraph->inputs().size());
  size_t idx = 0;
  for (auto input : subgraphNode->inputs()) {
    externalValuesMap[input] = subgraph->inputs()[idx];
    idx++;
  }

  for (size_t i = 0; i < subgraphNode->outputs().size(); ++i) {
    externalValuesMap[subgraphNode->outputs().at(i)] =
        subgraph->outputs().at(i);
  }

  // Add n's inputs to the group's input list if we don't already have them

  bool merging_node_after_subgraph = toMerge->isAfter(subgraphNode);
  Node* guard_node = merging_node_after_subgraph ? *subgraph->nodes().end()
                                                 : *subgraph->nodes().begin();
  WithInsertPoint guard(guard_node);

  std::unordered_set<Value*> closedValues =
      closedOverValues(toMerge, externalValuesMap);

  // There are currently downstream usage that relies on a fixed ordering
  // of graph inputs. TODO: remove
  std::vector<Value*> orderedClosedValues;
  std::unordered_set<Value*> orderedSeenValues;
  for (Value* input : toMerge->inputs()) {
    orderedClosedValues.push_back(input);
    orderedSeenValues.insert(input);
  }
  for (Value* closedValue : closedValues) {
    if (!orderedSeenValues.count(closedValue)) {
      orderedClosedValues.push_back(closedValue);
      orderedSeenValues.insert(closedValue);
    }
  }

  for (auto input : orderedClosedValues) {
    if (externalValuesMap.count(input) == 0) {
      // Clone constants inside the subgraph instead of referencing them, to
      // enable more optimizations
      if (auto value = toIValue(input)) {
        auto nv = subgraph->insertConstant(*value);
        nv->copyMetadata(input);
        externalValuesMap[input] = nv;
      } else {
        // The common case: this is a regular input, so just register it with
        // the group node and inner subgraph
        subgraphNode->addInput(input);
        auto inputToGraph = subgraph->addInput();
        inputToGraph->copyMetadata(input);
        externalValuesMap[input] = inputToGraph;
      }
    }
  }

  // Merge the node into the graph
  auto mergedNode = subgraph->insertNode(subgraph->createClone(
      toMerge, [&](Value* v) { return externalValuesMap[v]; }));

  if (!merging_node_after_subgraph) {
    // If n's outputs were inputs to `group`, remove them since we just merged
    // n in.
    //
    // i.e.,
    // x = f(w); group(x, y, z) becomes group(w, y, z).
    // x, y, z = f(w); group(x, y, z) becomes group(w).
    auto inputs = subgraphNode->inputs();
    for (size_t i = 0; i < toMerge->outputs().size(); ++i) {
      auto it = std::find(inputs.begin(), inputs.end(), toMerge->outputs()[i]);
      if (it != inputs.end()) {
        size_t p = it - inputs.begin();
        subgraphNode->removeInput(p);
        subgraph->inputs()[p]->replaceAllUsesWith(mergedNode->outputs()[i]);
        subgraph->eraseInput(p);
      }
    }
  }

  // Add n's outputs to the group node and inner subgraph outputs.
  for (const auto i : c10::irange(toMerge->outputs().size())) {
    auto oldOutput = toMerge->outputs()[i];
    auto newOutput = mergedNode->outputs()[i];
    subgraph->registerOutput(newOutput);
    auto groupOutput = subgraphNode->addOutput();
    groupOutput->copyMetadata(oldOutput);
    oldOutput->replaceAllUsesWith(groupOutput);
  }
  // Remove the original node now that the merge is complete
  if (destroyNode) {
    toMerge->destroy();
  }

  // We wait till destroying `toMerge` before pruning subgraph outputs,
  // since destroying `toMerge` could cause a subgraph output to no longer
  // have any uses
  const auto hasUsesOutsideSubgraph = [&](Value* v) {
    return std::any_of(
        v->uses().cbegin(), v->uses().cend(), [&](const Use& use) {
          return use.user->isAfter(subgraphNode);
        });
  };

  for (int64_t i = subgraphNode->outputs().size() - 1; i >= 0; i--) {
    if (!hasUsesOutsideSubgraph(subgraphNode->outputs().at(i))) {
      subgraphNode->eraseOutput(i);
      subgraph->eraseOutput(i);
    }
  }
}

Node* createSingletonSubgraph(Node* n, Symbol subgraphKind) {
  auto graph = n->owningGraph();
  auto subgraph = graph->create(subgraphKind, 0);
  subgraph->g_(attr::Subgraph, std::make_shared<Graph>(graph->current_scope()));
  subgraph->insertBefore(n);
  mergeNodeIntoSubgraph(n, subgraph);
  return subgraph;
}

void mergeNodeIntoSubgraphAndUpdateAliasing(
    Node* to_merge,
    Node* subgraphNode,
    AliasDb& db) {
  executeSubgraphMergeAndUpdateAliasing(to_merge, subgraphNode, db, [&]() {
    mergeNodeIntoSubgraph(to_merge, subgraphNode);
    return subgraphNode;
  });
}

Node* createSingletonSubgraphAndUpdateAliasing(
    Node* to_merge,
    Symbol subgraphKind,
    AliasDb& db) {
  return executeSubgraphMergeAndUpdateAliasing(
      to_merge, c10::nullopt, db, [&]() {
        return createSingletonSubgraph(to_merge, subgraphKind);
      });
}

bool unmergeOutputsAlisingInputs(Node* subgraphNode) {
  GRAPH_DEBUG("unfuseOutputsAlisingInputs on ", getHeader(subgraphNode));
  auto subgraph = subgraphNode->g(attr::Subgraph);
  AliasDb alias_db(subgraph);

  std::set<Node*, topo_cmp_node> nodes;
  for (auto o : subgraph->outputs()) {
    if (alias_db.mayContainAlias(o, subgraph->inputs())) {
      collectNodesToUnfuse(o->node(), nodes);
    }
  }

  // unfuse in the reverse topo order
  for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
    SubgraphUtils::unmergeNode(*it, subgraphNode);
  }

  return !nodes.empty();
}

bool unmergeAliasedOutputs(Node* subgraphNode) {
  GRAPH_DEBUG("unfuseAliasedOutputs on ", getHeader(subgraphNode));
  if (subgraphNode->outputs().size() < 2) {
    return false;
  }

  auto subgraph = subgraphNode->g(attr::Subgraph);
  GRAPH_DUMP("unfuseAliasedOutputs Subgraph ", subgraph);
  auto sets = buildAliasedSets(std::move(subgraph));
  GRAPH_DEBUG("buildAliasedSets sets.size() = ", sets.size());

  std::set<Node*, topo_cmp_node> nodes;

  for (auto i : c10::irange(sets.size())) {
    if (sets[i].size() <= 1) {
      GRAPH_DEBUG(
          "Set ",
          i,
          " with leader ",
          (*(sets[i].begin()))->debugName(),
          " size = ",
          sets[i].size());
      continue;
    }

    // we have at least two aliased outputs
    // we skip the earliest node w.r.t. the topo order
    // NB. after some nodes are unfused, the outputs of some other nodes
    // may become the outputs of the subgraph and alias the remaining ones
    // so we have to re-run this function until there are no more changes
    auto it = ++sets[i].begin();
    while (it != sets[i].end()) {
      GRAPH_DEBUG(
          "root aliased value ", (*it)->debugName(), " node ", *(*it)->node());
      collectNodesToUnfuse((*it)->node(), nodes);
      it++;
    }
  }

  // unfuse in the reverse topo order
  for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
    unmergeNode(*it, subgraphNode);
  }

  return !nodes.empty();
}

void unmergeNode(Node* n, Node* subgraphNode) {
  // collect output indices
  GRAPH_DEBUG("unfuseNode node ", getHeader(n));
  auto subgraph = n->owningGraph();

  std::set<Value*> node_outputs(n->outputs().begin(), n->outputs().end());
  std::set<size_t> output_indices;
  std::set<Value*> node_inputs(n->inputs().begin(), n->inputs().end());

  std::unordered_map<Value*, Value*> local_map;
  auto env = [&](Value* v) {
    auto it = local_map.find(v);
    if (it != local_map.end()) {
      return it->second;
    }
    TORCH_INTERNAL_ASSERT(
        false,
        "all inputs should've been mapped. Couldn't map %",
        v->debugName());
    return v;
  };

  for (auto i : c10::irange(subgraph->outputs().size())) {
    if (node_outputs.count(subgraph->outputs().at(i)) != 0) {
      output_indices.insert(i);
    }

    if (node_inputs.count(subgraph->outputs().at(i)) != 0) {
      GRAPH_DEBUG(
          "output %",
          subgraph->outputs().at(i)->debugName(),
          " is already subgraph's output");
      GRAPH_DEBUG(
          "Mapping %",
          subgraph->outputs().at(i)->debugName(),
          " to %",
          subgraphNode->outputs().at(i)->debugName());
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
      // of a constant node and we already cloned the constant
      // into the outer graph and mapped its outputs
      continue;
    }

    Value* sno = nullptr;
    if (ni->node()->kind() == prim::Constant) {
      auto copy = subgraphNode->owningGraph()->createClone(ni->node(), env);
      subgraphNode->owningGraph()->insertNode(copy);
      // in case we have a multi-output const, map the rest of the outputs
      // so when we get to clone `n`, `n`'s clone will use the outputs of this
      // constant clone
      for (auto i : c10::irange(n->outputs().size())) {
        GRAPH_DEBUG(
            "Mapping %",
            ni->node()->output(i)->debugName(),
            " to %",
            copy->output(i)->debugName());
        local_map[ni->node()->output(i)] = copy->output(i);
      }
    } else {
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

  n->destroy();
}

std::string truncateStrWithHash(const std::string& s, size_t maxlen) {
  if (s.size() <= maxlen) {
    return s;
  }
  std::string hash_str = c10::to_string(c10::hash<std::string>{}(s));
  // If hash-string plus '_' can fit into maxlen, then truncate the original
  // string correspondingly so that the final string with the hash included fits
  // into maxlen. If that's not possible, at least truncate the original string
  // to maxlen (and append the hash to it).
  size_t trunc_len =
      (maxlen > hash_str.size() + 1) ? (maxlen - hash_str.size() - 1) : maxlen;
  std::stringstream truncated;
  truncated << s.substr(0, trunc_len);
  truncated << "_" << hash_str;
  return truncated.str();
}

std::string generateNameForGraph(
    const std::shared_ptr<Graph>& graph,
    size_t maxlen,
    const std::string& prefix) {
  std::stringstream graph_name;
  graph_name << prefix;
  for (Node* node : graph->nodes()) {
    if (!node->kind().is_aten()) {
      continue;
    }
    graph_name << "_" << node->kind().toUnqualString();
  }
  return truncateStrWithHash(graph_name.str(), maxlen);
}

} // namespace SubgraphUtils
} // namespace jit
} // namespace torch
