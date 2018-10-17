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
//
// The basic scheme is: nodes are assigned positional indices within a signed
// 64-bit space. We leave 2^16 spaces in between each node so that nodes can
// be inserted between them in the order. If we ever run out of space between
// nodes, the index is rebuilt such that the nodes are "spread out" again.
class TopologicalIndex {
 public:
  TopologicalIndex(const Node* input, const Node* output)
      : TopologicalIndex(input, output, 1 << 16, INT64_MIN, INT64_MAX) {}

  // Constructor for tests only, so we can test boundary conditions
  TopologicalIndex(
      const Node* input,
      const Node* output,
      int64_t defaultInterval,
      int64_t lowerBound,
      int64_t upperBound)
      : input_(input),
        output_(output),
        defaultInterval_(defaultInterval),
        lowerBound_(lowerBound),
        upperBound_(upperBound) {
    const auto midPoint = (lowerBound_ + upperBound_) / 2;
    // Put the output at the midpoint, because it's the head/tail of the
    // circular linked list of nodes.
    nodeToIndex_[output_] = midPoint;
    indexToNode_[midPoint] = output_;

    // Put the input node somewhere way lower in the index
    const auto inputIndex = (lowerBound_ + midPoint) / 2;
    nodeToIndex_[input_] = inputIndex;
    indexToNode_[inputIndex] = input_;
  }

  TopologicalIndex(const TopologicalIndex&) = delete;
  void operator=(const TopologicalIndex&) = delete;

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

  // Insert `toInsert` after `insertPoint` in the topological index
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

  // Insert `toInsert` before `insertPoint` in the topological index
  void insertAfter(const Node* insertPoint, const Node* toInsert) {
    // Can't insert a node twice
    AT_ASSERT(nodeToIndex_.count(toInsert) == 0);

    if (insertPoint == output_) {
      // inserting AFTER the head/output node means PREPEND to the graph
      return insertAfter(input_, toInsert);
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

  // Erase `toErase` from the topological index
  void erase(const Node* toErase) {
    indexToNode_.erase(nodeToIndex_.at(toErase));
    nodeToIndex_.erase(toErase);
  }

 private:
  // If we run out of space between nodes we need to rebuild the index and
  // "spread out" the nodes again.
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

  // The node list is implemented as a circular linked list, with the output
  // node as the head/tail. Therefore, the index needs to treat "appends" to the
  // output as appends to the input.
  const Node* input_;
  const Node* output_;

  const int64_t defaultInterval_;
  const int64_t lowerBound_;
  const int64_t upperBound_;

  std::unordered_map<const Node*, int64_t> nodeToIndex_;
  std::map<int64_t, const Node*> indexToNode_;
};

} // namespace jit
} // namespace torch
