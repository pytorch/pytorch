#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
// class AliasTracker
//
// This class tracks the "A points to B" graph for all values, as well as
// wildcards and writes. It is used by AliasDb to provide a higher-level API.
class AliasTracker {
 public:
  // Returns true iff `v` is present in the alias set tracker.
  bool contains(const Value* v) const;

  // Does `n` write to `v` directly? (Does not consider aliases)
  bool writesTo(Node* n, const Value* v) const;

  // Whether `a` *may* point to `b`
  bool pointsTo(const Value* a, const Value* b) const;

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

  // Return all aliases of `v`. This is the full set of any other value that
  // *may* represent the same memory location.
  // NOTE: this does not consider wildcard values
  std::unordered_set<const Value*> getAliases(const Value* v) const;

  // Get all nodes that write to `v` or a value that may alias `v`.
  std::unordered_set<Node*> getWrites(const Value* v) const;

  // Functionally equivalent to getWrites().size() > 0, but with a
  // short-circuiting implementation to be faster.
  bool hasWriters(const Value* v) const;

  // Get all nodes that write to a wildcard value.
  const std::unordered_set<Node*>& getWildcardWriters() const {
    return wildcardWriters_;
  }

  void dump() const;

 private:
  enum class BfsDirection {
    POINTS_TO,
    POINTED_FROM,
    // Consider both pointer directions. The closure obtained from this
    // represents the whole "alias set" of a value.
    BOTH
  };
  // `Element` represents the vertex in the points-to graph. It has a 1:1
  // relationship with IR `Value`s.
  struct Element {
    const Value* value = nullptr;
    // All values that this value *may* point to. It's possible to have multiple
    // values that you might point to due to control flow/complex ops
    std::unordered_set<Element*> pointsTo;
    // Backreference to values that point to `this`
    std::unordered_set<Element*> pointedFrom;
    // Nodes that write to this specific value.
    std::unordered_set<Node*> writers;

    // Do a breadth-first search over the graph, starting at `this` and
    // traversing in the direction `dir`.`fn` will be run on each element.
    //
    // If `shortCircuit` is set, then if `fn` evaluates to true the search will
    // short-circuit and return true. You can use this to do existence checks
    // on the graph or whatever.
    template <typename Fn>
    bool bfs(Fn fn, BfsDirection dir, bool shortCircuit = false) const;
  };

  // Structure that owns all the element pointers. It's a map of
  //  raw pointer -> unique_ptr to facilitate easy queries
  std::unordered_map<Element*, std::unique_ptr<Element>> elements_;
  // Index to look up whatever element corresponds to that value.
  std::unordered_map<const Value*, Element*> map_;
  // All values that may point to a wildcard value.
  std::unordered_set<const Value*> wildcards_;
  // All nodes that write to a wildcard
  std::unordered_set<Node*> wildcardWriters_;
  size_t numWrites_ = 0;

  /**
   * Caching layer.
   */
  using set_id_t = size_t;
  bool useCache_ = true;
  mutable std::unordered_map<const Element*, std::unordered_set<set_id_t>>
      elementToSet_;
  mutable std::unordered_map<set_id_t, std::unordered_set<Node*>> setToWrites_;
  mutable bool cacheStale_ = true;
  mutable set_id_t lastId = 0;

  // Cache results in a way to make common queries constant time.
  void cache() const;
  bool hasWritersCached(const Value* v) const;
  std::unordered_set<Node*> getWritersCached(const Value* v) const;
  bool assignSet(const Element* el, set_id_t id) const;
  set_id_t getFreshId() const {
    return ++lastId;
  };
};

} // namespace jit
} // namespace torch
