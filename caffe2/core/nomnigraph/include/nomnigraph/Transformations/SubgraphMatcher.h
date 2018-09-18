#ifndef NOM_TRANFORMATIONS_SUBGRAPH_MATCHER_H
#define NOM_TRANFORMATIONS_SUBGRAPH_MATCHER_H

#include "caffe2/core/common.h"
#include "nomnigraph/Graph/Graph.h"

#include <functional>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace nom {

namespace matcher {

/**
 * MatchGraph is a graph of MatchNode.
 *
 * MatchNode needs a NodeMatchCriteria (a predicate for node matching) and
 * - includeInSubgraph: whether this node and nodes/edges reachable from it
 * should be included in the return matched subgraph (if the pattern matches).
 * This is useful in case we would like to specify a matching pattern but do not
 * want part of the pattern to be included in the returned subgraph.
 * - A count, which means we may want to match this node multiple times from its
 * incoming edges. The count can be unlimited (think about it as a regex star).
 * - If nonTerminal flag is set, it means we will not consider outgoing edges
 * from the node when doing subgraph matching.
 */

template <typename NodeMatchCriteria>
class MatchNode {
 public:
  static const int kStarCount = -1;

  MatchNode(const NodeMatchCriteria& criteria) : criteria_(criteria) {}

  MatchNode() = default;
  MatchNode(const MatchNode&) = default;
  MatchNode& operator=(const MatchNode&) = default;
  MatchNode(MatchNode&&) = default;

  NodeMatchCriteria getCriteria() const {
    return criteria_;
  }

  int getCount() const {
    return count_;
  }

  MatchNode<NodeMatchCriteria>& count(int count) {
    count_ = count;
    return *this;
  }

  MatchNode<NodeMatchCriteria>& starCount() {
    return count(kStarCount);
  }

  MatchNode<NodeMatchCriteria>& nonTerminal() {
    nonTerminal_ = true;
    return *this;
  }

  MatchNode<NodeMatchCriteria>& excludeFromSubgraph() {
    includeInSubgraph_ = false;
    return *this;
  }

  bool isNonTerminal() const {
    return nonTerminal_;
  }

  bool shouldIncludeInSubgraph() const {
    return includeInSubgraph_;
  }

 private:
  NodeMatchCriteria criteria_;
  int count_ = 1;
  bool includeInSubgraph_ = true;
  bool nonTerminal_ = false;
};

template <typename NodeMatchCriteria>
using MatchGraph = Graph<MatchNode<NodeMatchCriteria>>;

template <typename NodeMatchCriteria>
using MatchNodeRef = typename MatchGraph<NodeMatchCriteria>::NodeRef;

// TODO: Reuse convertToDotString once convertToDotString can work
// with subgraph.
template <typename NodeMatchCriteria>
std::string debugString(
    MatchNodeRef<NodeMatchCriteria> rootCriteriaRef,
    bool invertGraphTraversal) {
  std::ostringstream out;
  auto rootNode = rootCriteriaRef->data();
  out << "{rootCriteria = '" << rootNode.getCriteria() << "'";
  if (rootNode.getCount() != 1) {
    out << ", count = " << rootNode.getCount();
  }
  if (rootNode.isNonTerminal()) {
    out << ", nonTerminal = " << rootNode.isNonTerminal();
  }
  auto edges = invertGraphTraversal ? rootCriteriaRef->getInEdges()
                                    : rootCriteriaRef->getOutEdges();
  if (!edges.empty()) {
    out << ", childrenCriteria = [";
    for (auto& child : edges) {
      auto nextNode = invertGraphTraversal ? child->tail() : child->head();
      out << debugString<NodeMatchCriteria>(nextNode, invertGraphTraversal)
          << ", ";
    }
    out << "]";
  }
  out << "}";
  return out.str();
}

template <typename NodeMatchCriteria, typename GraphType>
class SubgraphMatchResult {
 public:
  // Map from match node to corresponding node in the graph to be scanned.
  using MatchNodeMap = std::unordered_map<
      MatchNodeRef<NodeMatchCriteria>,
      typename GraphType::NodeRef>;

  static SubgraphMatchResult<NodeMatchCriteria, GraphType> notMatched(
      const std::string& debugMessage) {
    return SubgraphMatchResult<NodeMatchCriteria, GraphType>(
        false, debugMessage);
  }

  static SubgraphMatchResult<NodeMatchCriteria, GraphType> notMatched() {
    return SubgraphMatchResult<NodeMatchCriteria, GraphType>(
        false, "Debug message is not enabled");
  }

  static SubgraphMatchResult<NodeMatchCriteria, GraphType> matched(
      bool ownSubgraph = false) {
    return SubgraphMatchResult<NodeMatchCriteria, GraphType>(
        true, "Matched", ownSubgraph);
  }

  bool isMatch() const {
    return isMatch_;
  }

  std::string getDebugMessage() const {
    return debugMessage_;
  }

  std::shared_ptr<typename GraphType::SubgraphType> getMatchedSubgraph() const {
    return matchedSubgraph_;
  }

  std::shared_ptr<MatchNodeMap> getMatchNodeMap() const {
    return matchNodeMap_;
  }

 private:
  SubgraphMatchResult(
      bool isMatch,
      const std::string& debugMessage,
      bool ownSubgraph = false)
      : isMatch_(isMatch),
        debugMessage_(debugMessage),
        matchedSubgraph_(
            ownSubgraph ? std::shared_ptr<typename GraphType::SubgraphType>(
                              new typename GraphType::SubgraphType())
                        : nullptr),
        matchNodeMap_(
            ownSubgraph ? std::shared_ptr<MatchNodeMap>(new MatchNodeMap())
                        : nullptr) {}

  const bool isMatch_;
  const std::string debugMessage_;
  const std::shared_ptr<typename GraphType::SubgraphType> matchedSubgraph_;
  const std::shared_ptr<MatchNodeMap> matchNodeMap_;
};

/*
 * Utilities for subgraph matching.
 */
template <
    typename GraphType,
    typename NodeMatchCriteria,
    typename NodeMatcherClass>
struct SubgraphMatcher {
  using SubgraphMatchResultType =
      SubgraphMatchResult<NodeMatchCriteria, GraphType>;

  using ReplaceGraphOperation = std::function<bool(
      GraphType&,
      typename GraphType::NodeRef,
      const SubgraphMatchResultType&)>;

  static bool isNodeMatch(
      typename GraphType::NodeRef node,
      const NodeMatchCriteria& criteria) {
    return NodeMatcherClass::isMatch(node, criteria);
  }

  // Check if there can be a subgraph that matches the given criteria that
  // is rooted at the given rootNode.
  // The flag invertGraphTraversal specify if we should follow out edges or
  // in edges. The default is true which is useful for a functional
  // intepretation of a dataflow graph.
  static SubgraphMatchResultType isSubgraphMatch(
      typename GraphType::NodeRef root,
      const MatchNodeRef<NodeMatchCriteria>& rootCriteriaRef,
      bool invertGraphTraversal = true,
      bool debug = false) {
    // Create a matched result that owns a matched subgraph object and pass
    // the subgraph object around to construct it during matching.
    auto matchedResult = SubgraphMatchResultType::matched(true);
    auto result = isSubgraphMatchInternal(
        matchedResult.getMatchNodeMap(),
        matchedResult.getMatchedSubgraph(),
        root,
        rootCriteriaRef,
        rootCriteriaRef->data().shouldIncludeInSubgraph(),
        invertGraphTraversal,
        debug);
    return result.isMatch() ? matchedResult : result;
  }

  // Utility to transform a graph by looking for subgraphs that match
  // a given pattern and then allow callers to mutate the graph based on
  // subgraphs that are found.
  // The current implementation doesn't handle any graph transformation
  // itself. Callers should be responsible for all intended mutation, including
  // deleting nodes in the subgraphs found by this algorithm.
  // Note: if the replaceFunction lambda returns false, the entire procedure
  // is aborted. This maybe useful in certain cases when we want to terminate
  // the subgraph search early.
  // invertGraphTraversal flag: see documentation in isSubgraphMatch
  static void replaceSubgraph(
      GraphType& graph,
      const MatchNodeRef<NodeMatchCriteria>& criteria,
      const ReplaceGraphOperation& replaceFunction,
      bool invertGraphTraversal = true) {
    for (auto nodeRef : graph.getMutableNodes()) {
      // Make sure the node is still in the graph.
      if (!graph.hasNode(nodeRef)) {
        continue;
      }
      auto matchResult =
          isSubgraphMatch(nodeRef, criteria, invertGraphTraversal);
      if (matchResult.isMatch()) {
        if (!replaceFunction(graph, nodeRef, matchResult)) {
          // If replaceFunction returns false, it means that we should abort
          // the entire procedure.
          break;
        }
      }
    }
  }

 private:
  static SubgraphMatchResultType isSubgraphMatchInternal(
      std::shared_ptr<typename SubgraphMatchResultType::MatchNodeMap>
          matchedNodes,
      std::shared_ptr<typename GraphType::SubgraphType> matchedSubgraph,
      typename GraphType::NodeRef root,
      const MatchNodeRef<NodeMatchCriteria>& rootCriteriaRef,
      bool includeInSubgraph,
      bool invertGraphTraversal,
      bool debug) {
    auto rootCriteriaNode = rootCriteriaRef->data();

    if (rootCriteriaNode.getCount() == 1) {
      auto matchedNodeEntry = matchedNodes->find(rootCriteriaRef);
      if (matchedNodeEntry != matchedNodes->end()) {
        // If rootCriteriaRef has been matched before (without multiplicity),
        // we should look up the corresponding matched node in the graph
        // and verify if it is the same.
        auto matchedNode = matchedNodeEntry->second;
        if (matchedNode == root) {
          return SubgraphMatchResultType::matched();
        } else if (debug) {
          std::ostringstream debugMessage;
          debugMessage << "Subgraph root at " << root << " is not the same as "
                       << matchedNode << " which previously matched criteria "
                       << debugString<NodeMatchCriteria>(
                              rootCriteriaRef, invertGraphTraversal);
          return SubgraphMatchResultType::notMatched(debugMessage.str());
        } else {
          return SubgraphMatchResultType::notMatched();
        }
      }
    }

    if (!isNodeMatch(root, rootCriteriaNode.getCriteria())) {
      if (debug) {
        std::ostringstream debugMessage;
        debugMessage << "Subgraph root at " << root
                     << " does not match criteria "
                     << debugString<NodeMatchCriteria>(
                            rootCriteriaRef, invertGraphTraversal);
        return SubgraphMatchResultType::notMatched(debugMessage.str());
      } else {
        return SubgraphMatchResultType::notMatched();
      }
    }
    if (rootCriteriaNode.isNonTerminal()) {
      // This is sufficient to be a match if this criteria specifies a non
      // terminal node.
      matchedNodes->emplace(rootCriteriaRef, root);
      if (includeInSubgraph) {
        matchedSubgraph->addNode(root);
      }
      return SubgraphMatchResultType::matched();
    }
    auto& edges =
        invertGraphTraversal ? root->getInEdges() : root->getOutEdges();

    int numEdges = edges.size();
    const auto criteriaEdges = invertGraphTraversal
        ? rootCriteriaRef->getInEdges()
        : rootCriteriaRef->getOutEdges();
    int numChildrenCriteria = criteriaEdges.size();

    // The current algorithm implies that the ordering of the children is
    // important. The children nodes will be matched with the children subgraph
    // criteria in the given order.

    int currentEdgeIdx = 0;
    for (int criteriaIdx = 0; criteriaIdx < numChildrenCriteria;
         criteriaIdx++) {
      auto childrenCriteriaRef = invertGraphTraversal
          ? criteriaEdges[criteriaIdx]->tail()
          : criteriaEdges[criteriaIdx]->head();

      int expectedCount = childrenCriteriaRef->data().getCount();
      bool isStarCount =
          expectedCount == MatchNode<NodeMatchCriteria>::kStarCount;

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
        bool shouldIncludeEdgeInSubgraph =
            childrenCriteriaRef->data().shouldIncludeInSubgraph() &&
            includeInSubgraph;

        if (!isSubgraphMatchInternal(
                 matchedNodes,
                 matchedSubgraph,
                 child,
                 childrenCriteriaRef,
                 shouldIncludeEdgeInSubgraph,
                 invertGraphTraversal,
                 debug)
                 .isMatch()) {
          if (!isStarCount) {
            // If the current criteria isn't a * pattern, this indicates a
            // failure.
            if (debug) {
              std::ostringstream debugMessage;
              debugMessage << "Child node at " << child
                           << " does not match child criteria "
                           << debugString<NodeMatchCriteria>(
                                  childrenCriteriaRef, invertGraphTraversal)
                           << ". We expected " << expectedCount
                           << " matches but only found " << countMatch << ".";
              return SubgraphMatchResultType::notMatched(debugMessage.str());
            } else {
              return SubgraphMatchResultType::notMatched();
            }
          } else {
            // Otherwise, we should move on to the next children criteria.
            break;
          }
        } else if (shouldIncludeEdgeInSubgraph) {
          matchedSubgraph->addEdge(edge);
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
                       << debugString<NodeMatchCriteria>(
                              childrenCriteriaRef, invertGraphTraversal)
                       << " but only found " << countMatch;
          return SubgraphMatchResultType::notMatched(debugMessage.str());
        } else {
          return SubgraphMatchResultType::notMatched();
        }
      }
    }

    if (currentEdgeIdx < numEdges) {
      // Fails because there are unmatched edges.
      if (debug) {
        std::ostringstream debugMessage;
        debugMessage << "Unmatched children for subgraph root at " << root
                     << ". There are " << numEdges
                     << " children, but only found " << currentEdgeIdx
                     << " matches for the children criteria.";
        return SubgraphMatchResultType::notMatched(debugMessage.str());
      } else {
        return SubgraphMatchResultType::notMatched();
      }
    }
    matchedNodes->emplace(rootCriteriaRef, root);
    if (includeInSubgraph) {
      matchedSubgraph->addNode(root);
    }
    return SubgraphMatchResultType::matched();
  }
};

} // namespace matcher

} // namespace nom

#endif // NOM_TRANFORMATIONS_SUBGRAPH_MATCHER_H
