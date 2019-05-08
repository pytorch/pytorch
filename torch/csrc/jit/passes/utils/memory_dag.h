#pragma once

#include <c10/util/ArrayRef.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {

struct Element;
struct Value;

// class MemoryDAG
//
// This class tracks the "A points to B" graph for all values. It is used by
// AliasDb to provide a higher-level API.
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
class MemoryDAG {
 public:
  // Make `from` point at `to`.
  TORCH_API void makePointerTo(Element* from, Element* to);

  void addToContainedElements(Element* contained, Element* container);

  // Make a fresh element (i.e. an element that doesn't point to anything) and
  // return it.
  TORCH_API Element* makeFreshValue(const Value* v);

  // Do `a` and `b` potentially share a memory location?
  bool mayAlias(const Element* a, const Element* b) const;
  TORCH_API bool mayAlias(Element* a, Element* b) const;

  // Does a hold reference to any memory that is stored in elem, or vice versa?
  bool mayContainAlias(const Element* a, const Element* b) const;
  bool mayContainAlias(Element* a, Element* b) const;

  bool mayContainAlias(
      const at::ArrayRef<Element*>& a,
      const at::ArrayRef<Element*>& b) const;

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
      const auto element = *it;

      for (const auto loc : element->getMemoryLocations()) {
        memoryLocations.insert(loc);
      }

      const auto cnt = a.count(*it);
      std::advance(it, cnt);
    }

    // If any of group `b`s memory locations overlap, return true.
    for (auto it = b.cbegin(); it != b.cend();) {
      const auto element = *it;

      for (const auto loc : element->getMemoryLocations()) {
        if (memoryLocations.count(loc)) {
          return true;
        }
      }

      const auto cnt = b.count(*it);
      std::advance(it, cnt);
    }
    // No overlap, so group `a` and `b` do not share a memory location
    return false;
  }

 private:
  bool memoryLocationOverlap(
      const std::unordered_set<const Element*>& a,
      const std::unordered_set<const Element*>& b) const;
  bool mayAliasImpl(const Element* a, const Element* b) const;
  bool mayContainAliasImpl(const Element* contained, const Element* container)
      const;
  // Structure that owns all the element pointers. It's a map of
  //  raw pointer -> unique_ptr to facilitate easy queries
  std::unordered_map<Element*, std::unique_ptr<Element>> elements_;
};

enum class BfsDirection {
  POINTS_TO,
  POINTED_FROM,
};

// `Element` represents the vertex in the points-to graph. It represents
// anything that could have an aliasing relationship, mostly IR `Value`s, but
// also the "inside of a list", or wildcards.
struct Element {
  // The value that this element corresponds to. May be null if this element
  // doesn't represent a first-class value.
  const Value* value = nullptr;

  // All elements that this element *may* point to. It's possible to have
  // multiple elements that you might point to due to control flow/complex ops
  std::unordered_set<Element*> pointsTo;
  // Backreference for points-to.
  std::unordered_set<Element*> pointedFrom;

  std::unordered_set<Element*> contained_elements;

  // Return the unique memory locations that `Element` might represent.
  TORCH_API std::unordered_set<const Element*> getMemoryLocations() const;
  // We do path compression to make repeated memory location queries faster.
  // An empty cache means it is invalidated (it can never be empty otherwise,
  // since every element must point to at least one memory location).
  mutable std::unordered_set<const Element*> cachedMemoryLocations_;

  // Do a breadth-first search over the graph, starting at `this` and
  // traversing in the direction `dir`.`fn` will be run on each element.
  template <typename Fn>
  bool bfs(Fn fn, BfsDirection dir) const;
};
} // namespace jit
} // namespace torch
