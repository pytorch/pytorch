//===- nomnigraph/Graph/Algorithms.h - Graph algorithms ---------*- C++ -*-===//
//
// TODO Licensing.
//
//===----------------------------------------------------------------------===//
//
// This file defines algorithms that only require Graph level annotations.
// Tarjans is defined.
//
//===----------------------------------------------------------------------===//

#ifndef NOM_GRAPH_ALGORITHMS_H
#define NOM_GRAPH_ALGORITHMS_H

#include <assert.h>
#include <unordered_map>
#include <unordered_set>

#include "nomnigraph/Graph/BinaryMatchImpl.h"
#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Graph/TarjansImpl.h"
#include "nomnigraph/Graph/TopoSort.h"

namespace nom {
namespace algorithm {

/// \brief Helper for dominator tree finding.
template <typename G>
void reachable(
    typename G::NodeRef root,
    typename G::NodeRef ignored,
    std::unordered_set<typename G::NodeRef>* seen) {
  seen->insert(root);
  for (const auto& outEdge : root->getOutEdges()) {
    auto& newNode = outEdge->head();
    if (newNode != ignored && (seen->find(newNode) == seen->end())) {
      reachable<G>(newNode, ignored, seen);
    }
  }
}

/// \brief A dominator tree finder.  Runs in O(M*N), there exist
/// more efficient implementations.
///
/// High level description of the algorithm:
///
/// 1) Find a map of {node}->{dominator set}
/// --
/// allNodes = reachable(root)
/// for n in nodes:
///   temporarily delete n from the graph
///   dom[n] = allNodes - reachable(root)
///   restore n to the graph
///
/// 2) Construct tree from that map
/// --
/// starting at root, BFS in dominatorMap:
///   if newnode has inedge, delete it
///   draw edge from parent to child
template <typename G>
Graph<typename G::NodeRef> dominatorTree(
    G* g,
    typename G::NodeRef source = nullptr) {
  assert(
      g->getMutableNodes().size() > 0 &&
      "Cannot find dominator tree of empty graph.");
  if (!source) {
    auto rootSCC = tarjans(g).back();
    assert(
        rootSCC.getNodes().size() == 1 &&
        "Cannot determine source node topologically, please specify one.");
    for (auto& node : rootSCC.getNodes()) {
      source = node;
      break;
    }
  }

  std::unordered_set<typename G::NodeRef> allNodes;
  Graph<typename G::NodeRef> tree;
  std::unordered_map<
      typename G::NodeRef,
      typename Graph<typename G::NodeRef>::NodeRef>
      mapToTreeNode;
  std::unordered_map<
      typename G::NodeRef,
      std::unordered_set<typename G::NodeRef>>
      dominatorMap;

  for (auto node : g->getMutableNodes()) {
    mapToTreeNode[node] = tree.createNode(std::move(node));
    if (node == source) {
      continue;
    }
    dominatorMap[source].insert(node);
  }

  for (const auto& node : g->getMutableNodes()) {
    if (node == source) {
      continue;
    }
    std::unordered_set<typename G::NodeRef> seen;
    std::unordered_set<typename G::NodeRef> dominated;
    reachable<G>(source, node, &seen);
    for (auto testNode : dominatorMap[source]) {
      if (seen.find(testNode) == seen.end() && testNode != node) {
        dominated.insert(testNode);
      }
    }
    dominatorMap[node] = dominated;
  }

  std::unordered_set<typename G::NodeRef> nextPass;
  nextPass.insert(source);

  while (nextPass.size()) {
    for (auto parent_iter = nextPass.begin(); parent_iter != nextPass.end();) {
      auto parent = *parent_iter;
      for (auto child : dominatorMap[parent]) {
        while (mapToTreeNode[child]->getInEdges().size()) {
          tree.deleteEdge(mapToTreeNode[child]->getInEdges().front());
        }
        tree.createEdge(mapToTreeNode[parent], mapToTreeNode[child]);
        if (dominatorMap.find(child) != dominatorMap.end()) {
          nextPass.insert(child);
        }
      }
      nextPass.erase(parent_iter++);
    }
  }

  return tree;
}

/// \brief Map all nodes in the graph to their immediate dominators.
template <typename G>
std::unordered_map<typename G::NodeRef, typename G::NodeRef>
immediateDominatorMap(G* g, typename G::NodeRef source = nullptr) {
  std::unordered_map<typename G::NodeRef, typename G::NodeRef> idomMap;
  auto idomTree = dominatorTree(g, source);
  for (auto node : idomTree.getMutableNodes()) {
    // Sanity check, really should never happen.
    assert(
        node->getInEdges().size() <= 1 &&
        "Invalid dominator tree generated from graph, cannot determing idom map.");
    // In degenerate cases, or for the root node, we self dominate.
    if (node->getInEdges().size() == 0) {
      idomMap[node->data()] = node->data();
    } else {
      auto idom = node->getInEdges()[0]->tail();
      idomMap[node->data()] = idom->data();
    }
  }
  return idomMap;
}

/// \brief Map all nodes to their dominance frontiers:
/// a set of nodes that does not strictly dominate the given node but does
/// dominate an immediate predecessor.  This is useful as it is the exact
/// location for the insertion of phi nodes in SSA representation.
template <typename G>
std::unordered_map<typename G::NodeRef, std::unordered_set<typename G::NodeRef>>
dominanceFrontierMap(G* g, typename G::NodeRef source = nullptr) {
  auto idomMap = immediateDominatorMap(g, source);
  std::unordered_map<
      typename G::NodeRef,
      std::unordered_set<typename G::NodeRef>>
      domFrontierMap;
  for (const auto node : g->getMutableNodes()) {
    if (node->getInEdges().size() < 2) {
      continue;
    }
    for (auto inEdge : node->getInEdges()) {
      auto predecessor = inEdge->tail();
      // This variable will track all the way up the dominator tree.
      auto runner = predecessor;
      while (runner != idomMap[node]) {
        domFrontierMap[runner].insert(node);
        runner = idomMap[runner];
      }
    }
  }
  return domFrontierMap;
}

/// \brief Induces edges on a subgraph by connecting all nodes
/// that are connected in the original graph.
template <typename SubgraphType>
void induceEdges(SubgraphType* sg) {
  for (auto& node : sg->getNodes()) {
    // We can scan only the inEdges
    for (auto& inEdge : node->getInEdges()) {
      if (sg->hasNode(inEdge->tail())) {
        sg->addEdge(inEdge);
      }
    }
  }
}

/// \brief Create subgraph object from graph.
template <typename GraphType>
typename GraphType::SubgraphType createSubgraph(GraphType* g) {
  typename GraphType::SubgraphType subgraph;
  for (auto& node : g->getMutableNodes()) {
    subgraph.addNode(node);
  }
  induceEdges(&subgraph);
  return subgraph;
}

} // namespace algorithm
} // namespace nom

#endif // NOM_GRAPH_ALGORITHMS_H
