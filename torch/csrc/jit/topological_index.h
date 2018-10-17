#pragma once

#include <ATen/core/Error.h>
#include <map>
#include <unordered_map>

namespace torch {
namespace jit {

struct Node;

// Index to track a topological ordering of nodes. Owned by a block,
// representing the nodes in the block.
//
// This lets us answer questions like "is this node before another node"
// efficiently, which is useful for optimization. It should be kept up to date
// with node insertions/deletions by the owning block.
class TopologicalIndex {
 public:
  TopologicalIndex(Node* tail)
      : TopologicalIndex(tail, 1 << 16, INT64_MIN, INT64_MAX) {}

  // Constructor for tests only, so we can test boundary conditions
  TopologicalIndex(
      Node* tail,
      int64_t defaultInterval,
      int64_t lowerBound,
      int64_t upperBound)
      : tail_(tail),
        defaultInterval_(defaultInterval),
        lowerBound_(lowerBound),
        upperBound_(upperBound) {
    nodeToIndex_[tail_] = 0;
    indexToNode_[0] = tail_;
  }
  // is `lhs` before `rhs`?
  bool isBefore(const Node* lhs, const Node* rhs) const {
    AT_ASSERT(lhs != rhs);
    return nodeToIndex_.at(lhs) < nodeToIndex_.at(rhs);
  }
  // is `lhs` after `rhs`?
  bool isAfter(const Node* lhs, const Node* rhs) const {
    AT_ASSERT(lhs != rhs);
    return nodeToIndex_.at(lhs) > nodeToIndex_.at(rhs);
  }

  void insertBefore(const Node* insertPoint, const Node* toInsert) {
    // Can't insert a node twice
    AT_ASSERT(nodeToIndex_.count(toInsert) == 0);

    auto indexIter = indexToNode_.find(nodeToIndex_[insertPoint]);
    const auto insertIndex = indexIter->first;
    if (indexIter == indexToNode_.begin()) {
      // Check we're not running off the end
      if (insertIndex < (lowerBound_ + defaultInterval_)) {
        reIndex();
        return insertBefore(insertPoint, toInsert);
      }

      // Move down a suitably vast distance and add the node
      const auto newIndex = nodeToIndex_[insertPoint] - defaultInterval_;
      nodeToIndex_[toInsert] = newIndex;
      indexToNode_[newIndex] = toInsert;
    } else {
      // We are between two nodes. Find the previous one
      indexIter--;
      const int64_t prevIndex = indexIter->first;
      const int64_t indexBetween = (insertIndex + prevIndex) / 2;
      if (indexToNode_.count(indexBetween) != 0) {
        reIndex();
        return insertBefore(insertPoint, toInsert);
      }

      nodeToIndex_[toInsert] = indexBetween;
      indexToNode_[indexBetween] = toInsert;
    }
  }

  void insertAfter(const Node* insertPoint, const Node* toInsert) {
    // Can't insert a node twice
    AT_ASSERT(nodeToIndex_.count(toInsert) == 0);

    if (insertPoint == tail_) {
      // inserting AFTER the head/tail node means PREPEND to the graph
      return insertBefore(getFirstNode(), toInsert);
    }

    auto indexIter = indexToNode_.find(nodeToIndex_[insertPoint]);
    const auto insertIndex = indexIter->first;
    // Are we at the end?
    if (indexIter->second != indexToNode_.rbegin()->second) {
      // if not, just insert before the next node
      insertBefore((++indexIter)->second, toInsert);
    } else {
      // check we're not running off the end
      if (insertIndex > (upperBound_ - defaultInterval_)) {
        reIndex();
        return insertAfter(insertPoint, toInsert);
      }

      // Move up a suitably vast distance and add the node
      const auto newIndex = nodeToIndex_[insertPoint] + defaultInterval_;
      nodeToIndex_[toInsert] = newIndex;
      indexToNode_[newIndex] = toInsert;
    }
  }

  void erase(const Node* toErase) {
    indexToNode_.erase(nodeToIndex_.at(toErase));
    nodeToIndex_.erase(toErase);
  }

 private:
  void reIndex() {
    AT_ASSERT(defaultInterval_ * (int64_t)nodeToIndex_.size() < upperBound_);

    std::map<int64_t, const Node*> newIndexToNode;
    std::unordered_map<const Node*, int64_t> newNodeToIndex;
    int64_t curPos = 0;
    for (const auto pr : indexToNode_) {
      newIndexToNode[curPos] = pr.second;
      newNodeToIndex[pr.second] = curPos;
      curPos += defaultInterval_;
    }
    indexToNode_.swap(newIndexToNode);
    nodeToIndex_.swap(newNodeToIndex);
  }

  const Node* getLastNode() const {
    return indexToNode_.rbegin()->second;
  }
  const Node* getFirstNode() const {
    return indexToNode_.begin()->second;
  }

  // The node list is implemented as a circular linked list.
  // We treat "appends" to the tail as prepends to the first node.
  const Node* tail_;

  const int64_t defaultInterval_;
  const int64_t lowerBound_;
  const int64_t upperBound_;

  std::unordered_map<const Node*, int64_t> nodeToIndex_;
  std::map<int64_t, const Node*> indexToNode_;
};

} // namespace jit
} // namespace torch
