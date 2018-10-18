#pragma once

#include <ATen/core/Error.h>
#include <map>
#include <unordered_map>

namespace torch {
namespace jit {

template <typename T>
struct TopologicalIndex;

struct Node;
using node_topological_index = TopologicalIndex<Node*>;
using topo_index_t = uint64_t;

// Index to track a topological ordering of nodes. Owned by a block,
// representing the nodes in the block.
//
// This lets us answer questions like "is this node before another node"
// efficiently, which is useful for optimization. It should be kept up to date
// with node insertions/deletions by the owning block.
//
// The basic scheme is: nodes are assigned topological indices within a
// 64-bit space. Appending a node moves assigns an index that's a big interval
// higher than the last node, giving room for insertions in between. If we ever
// run out of room, we rebuild the index.
//
// NOTE: this relies on some implementation details of the node listing, so
// don't reuse without checking those assumptions.
template <typename T>
struct TopologicalIndex {
 private:
  // The node list is implemented as a circular linked list, with the output
  // node as the head/tail. Therefore, the index needs to treat "appends" to the
  // output as appends to the input.
  T input_;
  T output_;

  // Lower and upper bounds of the index. Inclusive range.
  static constexpr topo_index_t lowerBound_ = 0;
  static constexpr topo_index_t upperBound_ = UINT64_MAX;

  // How far away to space nodes that are appended to the graph.
  // should be 2^n, where:
  //   - n is the maximum number of repeated insertions without a re-index
  //   - 2^(64-n) is the maximum number of appends to the end without reindex
  static constexpr topo_index_t defaultInterval_ = 1099511627776ULL; // 2^40

  std::map<topo_index_t, T> indexToObj_;

 public:
  // Constructor for tests only, so we can test boundary conditions
  TopologicalIndex(T input, T output) : input_(input), output_(output) {
    AT_ASSERT(upperBound_ > lowerBound_);

    setIndex(input_, lowerBound_);

    // Don't put the output node in the index, since that would prevent us from
    // appending efficiently. Instead just give it the max index for comparison
    output_->topo_index_ = upperBound_;
  }

  TopologicalIndex(const TopologicalIndex&) = delete;
  void operator=(const TopologicalIndex&) = delete;

  // Insert `toInsert` after `insertPoint` in the topological index
  void insertAfter(T insertPoint, T toInsert) {
    if (insertPoint == output_) {
      // see note on input_/output_
      return insertAfter(input_, toInsert);
    }

    auto indexIter = indexToObj_.find(insertPoint->topo_index_);
    AT_ASSERT(indexIter != indexToObj_.end());
    const auto insertIndex = indexIter->first;

    // Are we the last node?
    if (indexIter->second == indexToObj_.rbegin()->second) {
      // check if we're running off the end of the index
      if (insertIndex >= (upperBound_ - defaultInterval_)) {
        reIndex();
        return insertAfter(insertPoint, toInsert);
      }

      // Move down a suitably vast distance and add the node
      const auto newIndex = insertPoint->topo_index_ + defaultInterval_;
      setIndex(toInsert, newIndex);
    } else {
      // We're between two nodes, so insert between them.
      indexIter++;
      const topo_index_t nextIndex = indexIter->first;

      // Please mind integer overflow if changing this forumula
      const topo_index_t indexBetween =
          insertIndex + (nextIndex - insertIndex) / 2;

      if (indexToObj_.count(indexBetween) != 0) {
        // If we can't find a new spot, reindex and try again
        reIndex();
        return insertAfter(insertPoint, toInsert);
      }

      setIndex(toInsert, indexBetween);
    }
  }

 private:
  // update mappings of objs to the their topological index
  void setIndex(T obj, topo_index_t index) {
    indexToObj_[index] = obj;
    obj->topo_index_ = index;
  }

  // If we run out of space between nodes we need to rebuild the index and
  // "spread out" the nodes again.
  void reIndex() {
    AT_ASSERT(upperBound_ / defaultInterval_ > indexToObj_.size());
    std::map<topo_index_t, T> newIndexToObj;

    auto curIndex = lowerBound_;
    for (const auto pr : indexToObj_) {
      newIndexToObj[curIndex] = pr.second;
      pr.second->topo_index_ = curIndex;
      curIndex += defaultInterval_;
    }
    indexToObj_.swap(newIndexToObj);
  }
};
} // namespace jit
} // namespace torch
