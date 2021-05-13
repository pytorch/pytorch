#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/sparse_bitset.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/csrc/WindowsTorchApiMacro.h>

// Uses a compressed index representation for faster comparisons
typedef c10::SparseBitVector<256> MemoryLocations;
namespace torch {
namespace jit {

struct Element;
struct Value;
class MemoryDAG;

/**
 * Helper to build up the points-to graph.
 *
 * We separate the "building" into a different class because it allows us to
 * cache internally to MemoryDAG without worrying about how the DAG structure
 * is mutated.
 */
class TORCH_API MemoryDAGBuilder {
 public:
  // NOLINTNEXTLINE(modernize-use-equals-default)
  MemoryDAGBuilder() {}
  MemoryDAGBuilder(const MemoryDAGBuilder&) = delete;
  MemoryDAGBuilder& operator=(const MemoryDAGBuilder&) = delete;

  // Make `from` point at `to`.
  void makePointerTo(Element* from, Element* to);

  void addToContainedElements(Element* contained, Element* container);

  // Make a fresh element (i.e. an element that doesn't point to anything) and
  // return it.
  Element* makeFreshValue(const Value* v);

  friend MemoryDAG;

 private:
  std::vector<std::unique_ptr<Element>> indexToElementMap_;
};

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
class TORCH_API MemoryDAG {
 public:
  explicit MemoryDAG(std::unique_ptr<MemoryDAGBuilder> builder)
      : indexToElementMap_(std::move(builder->indexToElementMap_)) {}
  // explicitly delete copy constructor because otherwise windows build is
  // confused for an exported class see
  // https://stackoverflow.com/a/51033485/105137
  MemoryDAG(const MemoryDAG&) = delete;
  MemoryDAG& operator=(const MemoryDAG&) = delete;

  // Return the unique memory locations that `Element` might represent.
  const MemoryLocations& getMemoryLocations(const Element* e) const;

  // Do `a` and `b` potentially share a memory location?
  bool mayAlias(const Element* a, const Element* b) const;
  bool mayAlias(Element* a, Element* b) const;

  // Does a hold reference to any memory that is stored in elem, or vice versa?
  bool mayContainAlias(const Element* a, const Element* b) const;
  bool mayContainAlias(Element* a, Element* b) const;

  bool mayContainAlias(
      const at::ArrayRef<Element*> a,
      const at::ArrayRef<Element*> b) const;

  // Converts from the compressed index representation
  const Element* fromIndex(unsigned x) const;
  Element* fromIndex(unsigned x);
  void collectAllContainedMemoryLocations(
      const Element* elem,
      MemoryLocations& cont) const;

  /**
   * The following methods are special cases where we need to reach mutate the
   * internals of MemoryDAG for efficiency reasons. Don't call them unless you
   * know what you're doing! In particular, don't add new mutating methods
   * without ensuring that you are maintaining cache consistency for memory
   * locations.
   */
  // Adding wildcards can trigger extremely expensive cache invalidations. This
  // method adds them in a more efficient cache-aware way.
  void setWildcards(
      const std::unordered_set<const Value*>& wildcards,
      const ska::flat_hash_map<const Value*, Element*>& elementMap,
      const std::function<Element*(const Value*)>& getWildcardElement);
  Element* unsafeMakeFreshValue(const Value* v);

 private:
  bool mayAliasImpl(const Element* a, const Element* b) const;
  bool mayContainAliasImpl(const Element* contained, const Element* container)
      const;
  std::vector<std::unique_ptr<Element>> indexToElementMap_;
};

// `Element` represents the vertex in the points-to graph. It represents
// anything that could have an aliasing relationship, mostly IR `Value`s, but
// also the "inside of a list", or wildcards.
struct Element {
  Element(const Value* value_, unsigned index_);
  // wildcard constructor
  explicit Element(unsigned index_);

  // Index into the owning DAG's bit vector that represents this element.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  unsigned index;

  // All elements that this element *may* point to. It's possible to have
  // multiple elements that you might point to due to control flow/complex ops
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  MemoryLocations pointsTo;
  // Backreference for points-to.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  MemoryLocations pointedFrom;

  // Elements can contain other elements (e.g. List[Tensor])
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  MemoryLocations containedElements;

  // The values that this element corresponds to. May be empty if this element
  // doesn't represent a first-class value.
  // This is for debug information only.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unordered_set<const Value*> values;

 private:
  // Make `from` point at `to`.
  void makePointerTo(Element* from, Element* to);

  friend class MemoryDAG;
  // We memoize the results of `getMemoryLocations` to speed up queries.
  // A nullopt means that this cache is not yet populated. Since `MemoryDAG` is
  // immutable, this cache should never need to be invalidated.
  mutable c10::optional<MemoryLocations> cachedMemoryLocations_;
};

} // namespace jit
} // namespace torch
