#ifndef NOM_TRANFORMATIONS_SUBGRAPH_MATCHER_H
#define NOM_TRANFORMATIONS_SUBGRAPH_MATCHER_H

#include <functional>
#include <vector>

namespace nom {

namespace matcher {

/*
 * Subtree matching criteria consists of
 * - Node matching criteria for the subtree's root.
 * - Children subtree matching criteria
 * - A count, which means we may want more than one of this subtree. The count
 * can be unlimited. The count is only used when we match children of a subtree
 * root, not matching the subtree itself.
 * - If nonTerminal flag is set, it means we only match the root and do not
 * care about the children.
 */
template <typename NodeMatchCriteria>
class SubtreeMatchCriteria {
 public:
  static const int kStarCount = -1;
  SubtreeMatchCriteria(
      const NodeMatchCriteria& root,
      const std::vector<SubtreeMatchCriteria>& children = {},
      int count = 1,
      bool nonTerminal = false)
      : root_(root),
        children_(children),
        count_(count),
        nonTerminal_(nonTerminal){};

  // Non terminal
  static SubtreeMatchCriteria<NodeMatchCriteria> nonTerminal(
      const NodeMatchCriteria& root,
      int count = 1) {
    return SubtreeMatchCriteria(root, {}, count, true);
  }

 private:
  NodeMatchCriteria root_;
  std::vector<SubtreeMatchCriteria> children_;
  int count_;
  bool nonTerminal_;

  template <typename, typename, typename>
  friend class SubgraphMatcher;
};

/*
 * Utilities for subgraph matching.
 */
template <
    typename GraphType,
    typename NodeMatchCriteria,
    typename NodeMatcherClass>
struct SubgraphMatcher {
  static bool isNodeMatch(
      typename GraphType::NodeRef node,
      const NodeMatchCriteria& criteria) {
    return NodeMatcherClass::isMatch(node, criteria);
  }

  // Check if there can be a sub-tree that matches the given criteria that
  // is rooted at the given rootNode.
  // The flag invertGraphTraversal specify if we should follow out edges or
  // in edges. The default is true which is useful for a functional
  // intepretation of a dataflow graph.
  static bool isSubtreeMatch(
      typename GraphType::NodeRef root,
      const SubtreeMatchCriteria<NodeMatchCriteria>& criteria,
      bool invertGraphTraversal = true) {
    if (!isNodeMatch(root, criteria.root_)) {
      return false;
    }
    if (criteria.nonTerminal_) {
      // This is sufficient to be a match if this criteria specifies a non
      // terminal node.
      return true;
    }
    auto& edges =
        invertGraphTraversal ? root->getInEdges() : root->getOutEdges();

    int numEdges = edges.size();
    int numChildrenCriteria = criteria.children_.size();

    // The current algorithm implies that the ordering of the children is
    // important. The children nodes will be matched with the children subtree
    // criteria in the given order.

    int currentEdgeIdx = 0;
    for (int criteriaIdx = 0; criteriaIdx < numChildrenCriteria;
         criteriaIdx++) {
      auto childrenCriteria = criteria.children_[criteriaIdx];

      int expectedCount = childrenCriteria.count_;
      bool isStarCount =
          expectedCount == SubtreeMatchCriteria<NodeMatchCriteria>::kStarCount;

      int countMatch = 0;

      // Continue to match subsequent edges with the current children criteria.
      // Note that if the child criteria is a * pattern, this greedy algorithm
      // will attempt to find the longest possible sequence that matches the
      // children criteria.
      for (; currentEdgeIdx < numEdges &&
           (isStarCount || countMatch < expectedCount);
           currentEdgeIdx++) {
        auto edge = edges[currentEdgeIdx];
        auto child = invertGraphTraversal ? edge->tail() : edge->head();

        if (!isSubtreeMatch(child, childrenCriteria, invertGraphTraversal)) {
          if (!isStarCount) {
            // If the current criteria isn't a * pattern, this indicates a
            // failure.
            return false;
          } else {
            // Otherwise, we should move on to the next children criteria.
            break;
          }
        }

        countMatch++;
      }

      if (countMatch < expectedCount) {
        // Fails because there are not enough matches as specified by the
        // criteria.
        return false;
      }
    }

    if (currentEdgeIdx < numEdges) {
      // Fails because there are unmatched edges.
      return false;
    }
    return true;
  }

  // Utility to transform a graph by looking for subtrees that match
  // a given pattern and then allow callers to mutate the graph based on
  // subtrees that are found.
  // The current implementation doesn't handle any graph transformation
  // itself. Callers should be responsible for all intended mutation, including
  // deleting nodes in the subtrees found by this algorithm.
  // Note: if the replaceFunction lambda returns false, the entire procedure
  // is aborted. This maybe useful in certain cases when we want to terminate
  // the subtree search early.
  // invertGraphTraversal flag: see documentation in isSubtreeMatch
  static void replaceSubtree(
      GraphType& graph,
      const SubtreeMatchCriteria<NodeMatchCriteria>& criteria,
      const std::function<
          bool(GraphType& g, typename GraphType::NodeRef subtreeRoot)>&
          replaceFunction,
      bool invertGraphTraversal = true) {
    for (auto nodeRef : graph.getMutableNodes()) {
      // Make sure the node is still in the graph.
      if (!graph.hasNode(nodeRef)) {
        continue;
      }
      if (isSubtreeMatch(nodeRef, criteria, invertGraphTraversal)) {
        if (!replaceFunction(graph, nodeRef)) {
          // If replaceFunction returns false, it means that we should abort
          // the entire procedure.
          break;
        }
      }
    }
  }
};

} // namespace matcher

} // namespace nom

#endif // NOM_TRANFORMATIONS_SUBGRAPH_MATCHER_H
