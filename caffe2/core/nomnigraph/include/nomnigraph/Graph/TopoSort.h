#ifndef NOM_GRAPH_TOPO_SORT_H
#define NOM_GRAPH_TOPO_SORT_H

#include <unordered_map>

#include "nomnigraph/Graph/Graph.h"

namespace nom {
namespace algorithm {

/// \brief Topological sort using DFS.
///
/// This algorithm takes a Graph object and returns node references in
/// topological order.
template <typename GraphT>
class TopoSort {
 private:
  using NodeRefT = typename GraphT::NodeRef;

  GraphT* graph;

  /// \brief performs DFS from given node.
  //  Each node and edge is visited no more than once.
  //  Visited nodes are pushed into result vector after all children has been
  //  processed. Return true if cycle is detected, otherwise false.
  bool dfs(
      NodeRefT node,
      std::unordered_map<NodeRefT, int>& status,
      std::vector<NodeRefT>& nodes) {
    // mark as visiting
    status[node] = 1;
    for (const auto& outEdge : node->getOutEdges()) {
      auto& newNode = outEdge->head();
      int newStatus = status[newNode];
      if (newStatus == 0) {
        if (dfs(newNode, status, nodes)) {
          return true;
        }
      } else if (newStatus == 1) {
        // find a node being visited, cycle detected
        return true;
      }
      // ignore visited node
    }
    nodes.push_back(node);
    // mark as visited
    status[node] = 2;
    return false;
  }

 public:
  TopoSort(GraphT* graph) : graph(graph) {}

  struct Result {
    enum { OK, CYCLE } status;
    std::vector<NodeRefT> nodes;
  };

  Result run() {
    std::vector<NodeRefT> nodes;
    std::unordered_map<NodeRefT, int> status;
    for (auto& node : graph->getMutableNodes()) {
      if (!status[node]) {
        if (dfs(node, status, nodes)) {
          return {Result::CYCLE, {}};
        }
      }
    }
    std::reverse(nodes.begin(), nodes.end());
    return {Result::OK, nodes};
  }
};

//// \brief A function wrapper to infer the graph template parameters.
/// TODO change this to const GraphT& g
template <typename GraphT>
typename TopoSort<GraphT>::Result topoSort(GraphT* g) {
  TopoSort<GraphT> t(g);
  return t.run();
}

} // namespace algorithm
} // namespace nom
#endif // NOM_GRAPH_TOPO_SORT_H
