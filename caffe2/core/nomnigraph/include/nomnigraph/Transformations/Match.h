//=== nomnigraph/Transformations/Match.h - Graph matching utils -*- C++ -*-===//
//
// TODO Licensing.
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for matching subgraphs.
//
//===----------------------------------------------------------------------===//

#ifndef NOM_TRANFORMATIONS_MATCH_H
#define NOM_TRANFORMATIONS_MATCH_H

#include "nomnigraph/Graph/Algorithms.h"

#include <algorithm>
#include <vector>

namespace nom {

template <typename T>
struct NodeEqualityDefault {
  static bool equal(const T& a, const T& b) {
    return a->data() == b->data();
  }
};

template <
    typename G,
    typename EqualityClass = NodeEqualityDefault<typename G::NodeRef>>
class Match {
 public:
  using SubgraphType = typename G::SubgraphType;

  Match(G& g) : MatchGraph(g) {
    // First we sort both the matching graph topologically.
    // This could give us a useful anchor in the best case.
    auto result = nom::algorithm::topoSort(&MatchGraph);
    MatchNodeList = result.nodes;
  }

  std::vector<SubgraphType> recursiveMatch(
      typename G::NodeRef candidateNode,
      std::vector<typename G::NodeRef> stack,
      SubgraphType currentSubgraph) {
    if (EqualityClass::equal(stack.back(), candidateNode)) {
      currentSubgraph.addNode(candidateNode);

      // Base case
      if (stack.size() == MatchNodeList.size()) {
        return std::vector<SubgraphType>{currentSubgraph};
      }

      // Recurse and accumulate matches
      stack.emplace_back(MatchNodeList.at(stack.size()));

      std::vector<SubgraphType> matchingSubgraphs;
      for (auto outEdge : candidateNode->getOutEdges()) {
        for (auto subgraph :
             recursiveMatch(outEdge->head(), stack, currentSubgraph)) {
          matchingSubgraphs.emplace_back(subgraph);
        }
      }
      return matchingSubgraphs;
    }

    // No match here, early bailout
    return std::vector<SubgraphType>{};
  }

  std::vector<SubgraphType> match(G& g) {
    std::vector<SubgraphType> out;

    std::vector<typename G::NodeRef> stack;
    stack.emplace_back(MatchNodeList.front());

    // Try each node in the candidate graph as the anchor.
    for (auto n : g.getMutableNodes()) {
      for (auto subgraph : recursiveMatch(n, stack, SubgraphType())) {
        out.emplace_back(subgraph);
      }
    }

    return out;
  }

 private:
  G& MatchGraph;
  std::vector<typename G::NodeRef> MatchNodeList;
};

} // namespace nom

#endif // NOM_TRANFORMATIONS_MATCH_H
