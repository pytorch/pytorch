#pragma once

#include <c10/util/Exception.h>
#include <map>
#include <unordered_map>

namespace torch {
namespace jit {

template <typename T>
struct TopologicalIndex;

struct Node;
using node_topological_index = TopologicalIndex<Node*>;
using topo_position_t = uint64_t;

// Index to track a topological ordering of nodes. Owned by a block,
// representing the nodes in the block.
//
// This lets us answer questions like "is this node before another node"
// efficiently, which is useful for optimization. It should be kept up to date
// with node insertions/deletions by the owning block.
//
// The basic scheme is: nodes are assigned topological indices within a
// 64-bit space. Appending a node moves assigns a position that's a big interval
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
  const topo_position_t lowerBound_;
  const topo_position_t upperBound_;

  // How far away to space nodes that are appended to the graph.
  // should be 2^n, where:
  //   - n is the maximum number of repeated insertions without a re-index
  //   - 2^(64-n) is the maximum number of appends to the end without reindex
  const topo_position_t defaultInterval_;

  std::map<topo_position_t, T> positionToObj_;

 public:
  TopologicalIndex(T input, T output)
      : TopologicalIndex(
            input,
            output,
            0,
            UINT64_MAX,
            1099511627776ULL /* 2^40 */) {}

  // This constructor is for tests only, so we can test boundary conditions.
  TopologicalIndex(
      T input,
      T output,
      topo_position_t lowerBound,
      topo_position_t upperBound,
      topo_position_t defaultInterval)
      : input_(input),
        output_(output),
        lowerBound_(lowerBound),
        upperBound_(upperBound),
        defaultInterval_(defaultInterval) {
    AT_ASSERT(upperBound_ > lowerBound_);

    setPos(input_, lowerBound_);

    // Don't put the output node in the index, since that would prevent us from
    // appending efficiently. Instead just give it the max index for comparison
    output_->topo_position_ = upperBound_;
  }

  TopologicalIndex(const TopologicalIndex&) = delete;
  void operator=(const TopologicalIndex&) = delete;

  // Insert `toInsert` after `insertPoint` in the topological index
  void insertAfter(T insertPoint, T toInsert) {
    if (insertPoint == output_) {
      // see note on input_/output_
      return insertAfter(input_, toInsert);
    }

    auto indexIter = positionToObj_.find(insertPoint->topo_position_);
    AT_ASSERT(indexIter != positionToObj_.end());
    const auto insertPos = indexIter->first;

    // Are we the last node?
    if (indexIter->second == positionToObj_.rbegin()->second) {
      // check if we're running off the end of the index
      if (insertPos >= (upperBound_ - defaultInterval_)) {
        reIndex();
        return insertAfter(insertPoint, toInsert);
      }

      // Move down a suitably vast distance and add the node
      const auto newIndex = insertPoint->topo_position_ + defaultInterval_;
      setPos(toInsert, newIndex);
    } else {
      // We're between two nodes, so insert between them.
      indexIter++;
      const topo_position_t nextPos = indexIter->first;

      // Please mind integer overflow if changing this forumula
      const topo_position_t posBetween = insertPos + (nextPos - insertPos) / 2;

      if (positionToObj_.count(posBetween) != 0) {
        // If we can't find a new spot, reindex and try again
        reIndex();
        return insertAfter(insertPoint, toInsert);
      }

      setPos(toInsert, posBetween);
    }
  }

  void erase(T toErase)  {
    JIT_ASSERT(positionToObj_.count(toErase->topo_position_) != 0);
    positionToObj_.erase(toErase->topo_position_);
  }


 private:
  // update mappings of objs to the their topological index
  void setPos(T obj, topo_position_t pos) {
    positionToObj_[pos] = obj;
    obj->topo_position_ = pos;
  }

  // If we run out of space between nodes we need to rebuild the index and
  // "spread out" the nodes again.
  void reIndex() {
    AT_ASSERT(upperBound_ / defaultInterval_ > positionToObj_.size());
    std::map<topo_position_t, T> newIndexToObj;

    auto curPos = lowerBound_;
    for (const auto pr : positionToObj_) {
      newIndexToObj[curPos] = pr.second;
      pr.second->topo_position_ = curPos;
      curPos += defaultInterval_;
    }
    positionToObj_.swap(newIndexToObj);
  }
};
} // namespace jit
} // namespace torch
