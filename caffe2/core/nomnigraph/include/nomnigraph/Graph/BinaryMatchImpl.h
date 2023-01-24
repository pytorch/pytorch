#ifndef NOM_GRAPH_BINARYMATCHIMPL_H
#define NOM_GRAPH_BINARYMATCHIMPL_H

#include "nomnigraph/Graph/Graph.h"

namespace nom {
namespace algorithm {

/// \brief A binary graph matching algorithm based on Kahn's algorithm.
template <typename F, typename T, typename... U>
std::vector<Subgraph<T, U...>> binaryMatch(Graph<T, U...>* g, F condition) {
  using G = Graph<T, U...>;

  auto swappableCondition = [&](typename G::NodeRef m, bool match) {
    return match ? condition(m) : !condition(m);
  };

  auto edges = g->getMutableEdges();
  std::unordered_set<typename G::EdgeRef> edgeSet(edges.begin(), edges.end());

  // Topologically sorted matching subgraphs.
  std::vector<Subgraph<T, U...>> sortedNodes;

  // Find the initial frontier.
  std::vector<typename G::NodeRef> frontier;
  std::vector<typename G::NodeRef> nextFrontier;

  for (auto n : g->getMutableNodes()) {
    if (n->getInEdges().size() == 0) {
      if (condition(n)) {
        frontier.emplace_back(n);
      } else {
        nextFrontier.emplace_back(n);
      }
    }
  }

  auto stillHasInEdge = [&](typename G::NodeRef m) {
    for (auto inEdge : m->getInEdges()) {
      if (edgeSet.count(inEdge)) {
        return true;
      }
    }
    return false;
  };

  // This boolean will store which type of match we are looking for.
  // If true we are looking for the condition to return true,
  // if false we are looking for the condition to return false
  bool match = true;

  // Only if we currently have a frontier should we add a subgraph to the
  // vector of matches.
  if (frontier.size()) {
    sortedNodes.emplace_back();
  }

  // As long as there is a frontier we continue the algorithm.
  while (frontier.size() || nextFrontier.size()) {
    // Swap everything if we exhausted the current frontier.
    if (!frontier.size() && nextFrontier.size()) {
      frontier = nextFrontier;
      nextFrontier.clear();
      match = !match;
      if (match) {
        sortedNodes.emplace_back();
      }
    }

    // The main algorithm is inspired by
    // https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
    // originally written by @yinghai
    auto n = frontier.back();
    if (match) {
      sortedNodes.back().addNode(n);
    }
    frontier.pop_back();

    for (auto outEdge : n->getOutEdges()) {
      auto m = outEdge->head();
      if (!edgeSet.count(outEdge)) {
        continue;
      }
      edgeSet.erase(outEdge);

      if (!stillHasInEdge(m)) {
        if (swappableCondition(m, match)) {
          frontier.emplace_back(m);
        } else {
          nextFrontier.emplace_back(m);
        }
      }
    }
  }

  if (edgeSet.size()) {
    assert(
        0 &&
        "Invalid graph for Kahn's algorithm, cycle detected.  Please use Tarjans.");
  }

  return sortedNodes;
}

} // namespace algorithm
} // namespace nom

#endif // NOM_GRAPH_BINARYMATCHIMPL_H
