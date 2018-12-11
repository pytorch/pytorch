#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch {
namespace jit {
namespace SubgraphUtils {
namespace {
bool isSubgraphNodeKind(Symbol s) {
  return s == prim::DifferentiableGraph || s == prim::FusionGroup;
}

bool isSubgraphNodeKind(Node* n) {
  return isSubgraphNodeKind(n->kind());
}

// Combine the nodes in two subgraph together. The nodes will end up in
// `mergeTo`, and `mergeFrom` is destroyed.
void mergeSubgraph(Node* mergeTo, Node* mergeFrom) {
  const auto nodes = unmergeSubgraph(mergeFrom);
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    mergeNodeIntoSubgraph(*it, mergeTo);
  }
}
} // namespace

std::shared_ptr<Graph> getSubgraph(Node* n) {
  JIT_ASSERT(isSubgraphNodeKind(n));
  return n->g(attr::Subgraph);
}

std::vector<Node*> unmergeSubgraph(Node* subgraphNode) {
  JIT_ASSERT(subgraphNode->kind() == prim::DifferentiableGraph);
  auto outerGraph = subgraphNode->owningGraph();

  std::vector<Node*> temporary_nodes;
  auto subgraph = getSubgraph(subgraphNode);

  // Initialize a map of inner graph values to outer graph values
  std::unordered_map<const Value*, Value*> innerToOuter;
  const auto innerInputs = subgraph->inputs();
  const auto outerInputs = subgraphNode->inputs();
  for (size_t i = 0; i < innerInputs.size(); ++i) {
    innerToOuter[innerInputs[i]] = outerInputs[i];
  }

  // Clone all nodes
  for (auto inner : subgraph->nodes()) {
    Node* outer = outerGraph->createClone(
        inner, [&](Value* k) -> Value* { return innerToOuter.at(k); });
    outer->insertBefore(subgraphNode);
    temporary_nodes.emplace_back(outer);
    const auto innerOutputs = inner->outputs();
    const auto outerOutputs = outer->outputs();
    for (size_t i = 0; i < innerOutputs.size(); ++i) {
      innerToOuter[innerOutputs[i]] = outerOutputs[i];
    }
  }

  // Replace uses of group outputs and destroy the group
  const auto subgraphOutputs = subgraph->outputs();
  JIT_ASSERT(subgraphOutputs.size() >= subgraphNode->outputs().size());
  for (size_t i = 0; i < subgraphNode->outputs().size(); ++i) {
    const auto outerOutput = innerToOuter.at(subgraphOutputs[i]);
    subgraphNode->outputs()[i]->replaceAllUsesWith(outerOutput);
  }
  subgraphNode->destroy();

  return temporary_nodes;
}

void mergeNodeIntoSubgraph(Node* toMerge, Node* subgraphNode) {
  JIT_ASSERT(isSubgraphNodeKind(subgraphNode));
  if (isSubgraphNodeKind(toMerge)) {
    return mergeSubgraph(subgraphNode, toMerge);
  }

  auto subgraph = getSubgraph(subgraphNode);

  // Map from values in the surrounding graph to inputs in the subgraph
  std::unordered_map<Value*, Value*> inputsMap;

  JIT_ASSERT(subgraphNode->inputs().size() == subgraph->inputs().size());
  size_t idx = 0;
  for (auto input : subgraphNode->inputs()) {
    inputsMap[input] = subgraph->inputs()[idx];
    idx++;
  }

  // Add n's inputs to the group's input list if we don't already have them
  WithInsertPoint guard(*subgraph->nodes().begin());
  for (auto input : toMerge->inputs()) {
    if (inputsMap.count(input) == 0) {
      // Clone constants inside the subgraph instead of referencing them, to
      // enable more optimizations
      if (auto value = toIValue(input)) {
        auto nv = subgraph->insertConstant(*value);
        inputsMap[input] = nv;
      } else {
        // The common case: this is a regular input, so just register it with
        // the group node and inner subgraph
        subgraphNode->addInput(input);
        auto inputToGraph = subgraph->addInput();
        inputToGraph->setType(input->type());
        inputsMap[input] = inputToGraph;
      }
    }
  }

  // Merge the node into the graph
  auto mergedNode = subgraph->insertNode(
      subgraph->createClone(toMerge, [&](Value* v) { return inputsMap[v]; }));

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

  // Add n's outputs to the group node and inner subgraph outputs.
  for (size_t i = 0; i < toMerge->outputs().size(); i++) {
    auto oldOutput = toMerge->outputs()[i];

    // Only register the output in the group node if it's actually used
    // outside the subgraph.
    const auto hasUsesOutsideSubgraph = std::any_of(
        oldOutput->uses().cbegin(),
        oldOutput->uses().cend(),
        [&](const Use& use) { return use.user->isAfter(subgraphNode); });

    if (hasUsesOutsideSubgraph) {
      auto newOutput = mergedNode->outputs()[i];
      subgraph->registerOutput(newOutput);
      auto groupOutput = subgraphNode->addOutput();
      groupOutput->copyMetadata(oldOutput);
      oldOutput->replaceAllUsesWith(groupOutput);
    }
  }

  // Remove the original node now that the merge is complete
  toMerge->destroy();
}

Node* createSingletonSubgraph(Node* n, Symbol subgraphKind) {
  JIT_ASSERT(isSubgraphNodeKind(subgraphKind));
  auto graph = n->owningGraph();
  auto subgraph = graph->create(subgraphKind, 0);
  subgraph->g_(attr::Subgraph, std::make_shared<Graph>(graph->current_scope()));
  subgraph->insertBefore(n);
  mergeNodeIntoSubgraph(n, subgraph);
  return subgraph;
}
} // namespace SubgraphUtils
} // namespace jit
} // namespace torch
