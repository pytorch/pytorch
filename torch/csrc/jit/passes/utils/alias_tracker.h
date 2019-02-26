#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

// class AliasTracker
//
// This class tracks the "A points to B" graph for all values, as well as
// wildcards and writes. It is used by AliasDb to provide a higher-level API.
//
// We maintain a DAG where:
//   - Vertices (called "elements") represent values and
//     other aliasing entities (e.g. like the stuff inside a list)
//   - Edges represent a "points-to" relationship.
//
// Leaves in this DAG are entities that don't point to anything, and thus
// correspond to unique "memory locations".
//
// So, by traversing the "points-to" graph to the leaves, you can determine
// which memory locations an element may point to.
class AliasTracker {
 public:
  // Returns true iff `v` is present in the alias set tracker.
  bool contains(const Value* v) const;

  // Does `n` write to a memory location that `v` may point to?
  bool writesTo(Node* n, const Value* v) const;

  // Make `v` point at `to`.
  void makePointerTo(const Value* v, const Value* to);

  // Give `v` a fresh alias (i.e. it does not point to any value)
  void makeFreshValue(const Value* v);

  // Register `v` as a wildcard value.
  void setWildcard(const Value* v);

  // is `v` a wildcard?
  bool isWildcard(const Value* v) const;

  // Register the fact that `n` writes to `v`.
  void registerWrite(const Value* v, Node* n);

  // Does anything write to the memory locations that `v` may point to?
  bool hasWriters(const Value* v) const;

  // Get all nodes that write to a wildcard value.
  const std::unordered_set<Node*>& getWildcardWriters() const {
    return wildcardWriters_;
  }

  // Get the values that represent the memory locations that `v` may point to.
  // Return values are guaranteed to be "fresh" tensors--they do not point to
  // anything else.
  ValueSet getMemoryLocations(const Value* v) const;

  // Do `a` and `b` potentially share a memory location?
  bool mayAlias(const Value* a, const Value* b) const;

  // Do any values in group `a` potentially share a memory location with any
  // value in group `b`?
  //
  // This is written so that either of the inputs could be a multiset
  template <typename T, typename U>
  bool mayAlias(const T& a, const U& b) const {
    if (a.empty() || b.empty()) {
      return false;
    }

    // Record all memory locations from group `a`
    std::unordered_set<const Element*> memoryLocations;
    for (auto it = a.cbegin(); it != a.cend();) {
      const auto value = *it;
      if (isWildcard(value)) {
        return true;
      }

      if (map_.count(value)) {
        for (const auto loc : map_.at(value)->getMemoryLocations()) {
          memoryLocations.insert(loc);
        }
      }

      const auto cnt = a.count(*it);
      std::advance(it, cnt);
    }

    // If any of group `b`s memory locations overlap, return true.
    for (auto it = b.cbegin(); it != b.cend();) {
      const auto value = *it;
      if (isWildcard(value)) {
        return true;
      }

      if (map_.count(value)) {
        for (const auto loc : map_.at(value)->getMemoryLocations()) {
          if (memoryLocations.count(loc)) {
            return true;
          }
        }
      }

      const auto cnt = b.count(*it);
      std::advance(it, cnt);
    }
    // No overlap, so group `a` and `b` do not share a memory location
    return false;
  }

  void dump() const;

 private:
  enum class BfsDirection {
    POINTS_TO,
    POINTED_FROM,
  };
  // `Element` represents the vertex in the points-to graph. It has a 1:1
  // relationship with IR `Value`s.
  struct Element {
    const Value* value = nullptr;
    // All elements that this element *may* point to. It's possible to have
    // multiple elements that you might point to due to control flow/complex ops
    std::unordered_set<Element*> pointsTo;
    // Backreference for points-to.
    std::unordered_set<Element*> pointedFrom;

    std::unordered_set<const Element*> getMemoryLocations() const;
    // We do path compression to make repeated memory location queries faster.
    // An empty cache means it is invalidated (it can never be empty otherwise,
    // since every element must point to at least one memory location).
    mutable std::unordered_set<const Element*> cachedMemoryLocations_;

    // Do a breadth-first search over the graph, starting at `this` and
    // traversing in the direction `dir`.`fn` will be run on each element.
    template <typename Fn>
    bool bfs(Fn fn, BfsDirection dir) const;
  };

  // Structure that owns all the element pointers. It's a map of
  //  raw pointer -> unique_ptr to facilitate easy queries
  std::unordered_map<Element*, std::unique_ptr<Element>> elements_;
  // Index to look up whatever element corresponds to that value.
  std::unordered_map<const Value*, Element*> map_;
  // All values that may point to a wildcard value.
  ValueSet wildcards_;
  // All nodes that write to a wildcard
  std::unordered_set<Node*> wildcardWriters_;
  size_t numWrites_ = 0;

  std::unordered_map<Node*, ValueSet> writeIndex_;
  mutable std::unordered_set<const Element*> writeCache_;
  mutable bool isWriteCacheStale_ = true;
  void rebuildWriteCache() const;
};

} // namespace jit
} // namespace torch
