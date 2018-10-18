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
using topo_index_t = int64_t;

// Index to track a topological ordering of nodes. Owned by a block,
// representing the nodes in the block.
//
// This lets us answer questions like "is this node before another node"
// efficiently, which is useful for optimization. It should be kept up to date
// with node insertions/deletions by the owning block.
//
// The basic scheme is: nodes are assigned topological indices within a signed
// 64-bit space. The upper and lower bounds of the space are assigned to the
// inputs and output nodes, respectively. Insertion just finds the midpoint
// between the two nearest nodes and places the node there.
template <typename T>
struct TopologicalIndex {
 public:
  TopologicalIndex(T input, T output)
      : TopologicalIndex(input, output, INT64_MIN, INT64_MAX) {}

  // Constructor for tests only, so we can test boundary conditions
  TopologicalIndex(
      T input,
      T output,
      topo_index_t lowerBound,
      topo_index_t upperBound)
      : input_(input),
        output_(output),
        lowerBound_(lowerBound),
        upperBound_(upperBound) {
    AT_ASSERT(upperBound_ > lowerBound_);
    setIndex(input_, lowerBound_);
    setIndex(output_, upperBound_);
  }

  TopologicalIndex(const TopologicalIndex&) = delete;
  void operator=(const TopologicalIndex&) = delete;

  // Insert `toInsert` before `insertPoint` in the topological index
  void insertAfter(T insertPoint, T toInsert) {
    if (insertPoint == output_) {
      // inserting after the head/output node means prepend to the graph by
      // inserting after the intput node
      return insertAfter(input_, toInsert);
    }

    auto indexIter = indexToObj_.find(insertPoint->topo_index_);
    const auto insertIndex = indexIter->first;
    const int64_t nextIndex = (++indexIter)->first;
    const int64_t indexBetween = (insertIndex + nextIndex) / 2;

    if (indexToObj_.count(indexBetween) != 0) {
      // If we can't fit a new spot, reindex and try again
      reIndex();
      return insertAfter(insertPoint, toInsert);
    }

    setIndex(toInsert, indexBetween);
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
    const auto spaceBetweenNodes =
        (upperBound_ - lowerBound_) / indexToObj_.size();
    std::map<int64_t, T> newIndexToObj;

    topo_index_t curPos = lowerBound_;
    for (const auto pr : indexToObj_) {
      newIndexToObj[curPos] = pr.second;
      pr.second->topo_index_ = curPos;

      curPos += spaceBetweenNodes;
    }

    indexToObj_.swap(newIndexToObj);

    // for consistency, put output_ at the upper bound of the range
    setIndex(output_, upperBound_);
  }

  // The node list is implemented as a circular linked list, with the output
  // node as the head/tail. Therefore, the index needs to treat "appends" to the
  // output as appends to the input.
  T input_;
  T output_;

  const int64_t lowerBound_;
  const int64_t upperBound_;

  std::map<int64_t, T> indexToObj_;
};

} // namespace jit
} // namespace torch
