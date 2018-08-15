#ifndef NOM_TRANFORMATIONS_SUBGRAPH_MATCHER_H
#define NOM_TRANFORMATIONS_SUBGRAPH_MATCHER_H

#include <functional>
#include <sstream>
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

  std::string debugString() const {
    std::ostringstream out;
    out << "{rootCriteria = '" << root_ << "'";
    if (count_ != 1) {
      out << ", count = " << count_;
    }
    if (nonTerminal_) {
      out << ", nonTerminal = " << nonTerminal_;
    }
    if (!children_.empty()) {
      out << ", childrenCriteria = [";
      for (auto& child : children_) {
        out << child.debugString() << ", ";
      }
      out << "]";
    }
    out << "}";
    return out.str();
  }

 private:
  NodeMatchCriteria root_;
  std::vector<SubtreeMatchCriteria> children_;
  int count_;
  bool nonTerminal_;

  template <typename, typename, typename>
  friend class SubgraphMatcher;
};

template <typename GraphType>
class SubtreeMatchResult {
 public:
  static SubtreeMatchResult<GraphType> notMatched(
      const std::string& debugMessage) {
    return SubtreeMatchResult<GraphType>(false, debugMessage);
  }

  static SubtreeMatchResult<GraphType> notMatched() {
    return SubtreeMatchResult<GraphType>(false, "Debug message is not enabled");
  }

  static SubtreeMatchResult<GraphType> matched() {
    return SubtreeMatchResult<GraphType>(true, "");
  }

  bool isMatch() const {
    return isMatch_;
  }

  std::string getDebugMessage() const {
    return debugMessage_;
  }

 private:
  SubtreeMatchResult(bool isMatch, const std::string& debugMessage)
      : isMatch_(isMatch), debugMessage_(debugMessage) {}

  const bool isMatch_;
  const std::string debugMessage_;
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
  static SubtreeMatchResult<GraphType> isSubtreeMatch(
      typename GraphType::NodeRef root,
      const SubtreeMatchCriteria<NodeMatchCriteria>& criteria,
      bool invertGraphTraversal = true,
      bool debug = false) {
    if (!isNodeMatch(root, criteria.root_)) {
      if (debug) {
        std::ostringstream debugMessage;
        debugMessage << "Subtree root at " << root
                     << " does not match criteria " << criteria.debugString();
        return SubtreeMatchResult<GraphType>::notMatched(debugMessage.str());
      } else {
        return SubtreeMatchResult<GraphType>::notMatched();
      }
    }
    if (criteria.nonTerminal_) {
      // This is sufficient to be a match if this criteria specifies a non
      // terminal node.
      return SubtreeMatchResult<GraphType>::matched();
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

        if (!isSubtreeMatch(child, childrenCriteria, invertGraphTraversal)
                 .isMatch()) {
          if (!isStarCount) {
            // If the current criteria isn't a * pattern, this indicates a
            // failure.
            if (debug) {
              std::ostringstream debugMessage;
              debugMessage << "Child node at " << child
                           << " does not match child criteria "
                           << childrenCriteria.debugString() << ". We expected "
                           << expectedCount << " matches but only found "
                           << countMatch << ".";
              return SubtreeMatchResult<GraphType>::notMatched(
                  debugMessage.str());
            } else {
              return SubtreeMatchResult<GraphType>::notMatched();
            }
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
        if (debug) {
          std::ostringstream debugMessage;
          debugMessage << "Expected " << expectedCount
                       << " matches for child criteria "
                       << childrenCriteria.debugString() << " but only found "
                       << countMatch;
          return SubtreeMatchResult<GraphType>::notMatched(debugMessage.str());
        } else {
          return SubtreeMatchResult<GraphType>::notMatched();
        }
      }
    }

    if (currentEdgeIdx < numEdges) {
      // Fails because there are unmatched edges.
      if (debug) {
        std::ostringstream debugMessage;
        debugMessage << "Unmatched children for subtree root at " << root
                     << ". There are " << numEdges
                     << " children, but only found " << currentEdgeIdx
                     << " matches for the children criteria.";
        return SubtreeMatchResult<GraphType>::notMatched(debugMessage.str());
      } else {
        return SubtreeMatchResult<GraphType>::notMatched();
      }
    }
    return SubtreeMatchResult<GraphType>::matched();
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
      if (isSubtreeMatch(nodeRef, criteria, invertGraphTraversal).isMatch()) {
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
