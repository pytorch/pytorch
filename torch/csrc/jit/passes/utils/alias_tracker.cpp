#include "alias_tracker.h"

#include <torch/csrc/utils/memory.h>
#include <queue>

namespace torch {
namespace jit {

// Returns true iff `v` is present in the alias set tracker.
bool AliasTracker::contains(const Value* v) const {
  return map_.count(v);
}

bool AliasTracker::writesTo(Node* n, const Value* v) const {
  if (isWildcard(v)) {
    return wildcardWriters_.count(n);
  }

  if (!map_.count(v)) {
    return false;
  }

  return map_.at(v)->writers.count(n);
}

// Whether `a` *may* point to `b`
bool AliasTracker::pointsTo(const Value* a, const Value* b) const {
  if (!map_.count(a)) {
    return false;
  }
  if (isWildcard(a) || isWildcard(b)) {
    return true;
  }

  // BFS the subtree where the root is `a`s element and the branches are the
  // `pointsTo` relationships.
  const auto root = map_.at(a);
  return root->bfs(
      [&](const Element* el) { return el->value == b; },
      BfsDirection::POINTS_TO,
      /*shortCircuit=*/true);
}

// Make `v` point at `to`.
void AliasTracker::makePointerTo(const Value* v, const Value* to) {
  if (v == to) {
    return;
  }

  // If `to` is a wildcard, don't insert anything into the graph; wildcards
  // are tracked separately since they have different aliasing rules.
  if (isWildcard(to)) {
    setWildcard(v);
    return;
  }

  if (!map_.count(to)) {
    makeFreshValue(to);
  }

  if (!map_.count(v)) {
    makeFreshValue(v);
  }

  auto vEl = map_.at(v);
  auto toEl = map_.at(to);

  vEl->pointsTo.insert(toEl);
  toEl->pointedFrom.insert(vEl);
}

// Give `v` a fresh alias (i.e. it does not point to any value)
void AliasTracker::makeFreshValue(const Value* v) {
  auto el = torch::make_unique<Element>();
  el->value = v;

  auto rawPtr = el.get();
  elements_.emplace(rawPtr, std::move(el));
  map_.emplace(v, rawPtr);
}

// Register `v` as a wildcard value.
void AliasTracker::setWildcard(const Value* v) {
  wildcards_.insert(v);
}

// is `v` a wildcard?
bool AliasTracker::isWildcard(const Value* v) const {
  return wildcards_.count(v);
}

// Register the fact that `n` writes to `v`.
void AliasTracker::registerWrite(const Value* v, Node* n) {
  numWrites_++;

  if (isWildcard(v)) {
    wildcardWriters_.insert(n);
    return;
  }

  AT_ASSERT(map_.count(v));
  map_.at(v)->writers.insert(n);
}

// Return all aliases of `v`. This is the full set of any other value that
// *may* represent the same memory location.
// NOTE: this does not consider wildcard values
std::unordered_set<const Value*> AliasTracker::getAliases(
    const Value* v) const {
  std::unordered_set<const Value*> ret;
  if (!map_.count(v)) {
    return ret;
  }

  const auto root = map_.at(v);

  root->bfs(
      [&](const Element* el) {
        ret.insert(el->value);
        return false; // fn has to return bool but we don't use the result
      },
      BfsDirection::BOTH);
  return ret;
}

// Get all nodes that write to `v` or a value that may alias `v`.
std::unordered_set<Node*> AliasTracker::getWrites(const Value* v) const {
  std::unordered_set<Node*> ret;
  if (!map_.count(v)) {
    return ret;
  }

  // Any write to a wilcard may write to `v`.
  for (auto writer : wildcardWriters_) {
    ret.insert(writer);
  }

  if (useCache_) {
    for (auto writer : getWritersCached(v)) {
      ret.insert(writer);
    }
    return ret;
  }

  const auto root = map_.at(v);
  root->bfs(
      [&](const Element* el) {
        for (auto writer : el->writers) {
          ret.insert(writer);
        }
        return false; // fn has to return bool but we don't use the result
      },
      BfsDirection::BOTH);

  return ret;
}

// Functionally equivalent to getWrites().size() > 0, but with a
// short-circuiting implementation to be faster.
bool AliasTracker::hasWriters(const Value* v) const {
  if (!map_.count(v)) {
    return false;
  }

  if (isWildcard(v)) {
    // If `n` has a wildcard, any write in the graph may write to it.
    // So the only way we know there are no writers is if there are no writes
    // at all.
    return numWrites_ == 0;
  }

  if (wildcardWriters_.size() > 0) {
    // A write to the wildcard may be a write to any value.
    return true;
  }

  if (useCache_) {
    return hasWritersCached(v);
  }

  const auto root = map_.at(v);
  return root->bfs(
      [&](const Element* el) { return el->writers.size() > 0; },
      BfsDirection::BOTH,
      /*shortCircuit=*/true);
}

void AliasTracker::dump() const {
  std::cout << "\n===2. ALIAS DB===\n";
  for (const auto& ptrPair : elements_) {
    const auto element = ptrPair.first;
    if (element->pointsTo.size() > 0) {
      std::cout << element->value->uniqueName() << " points to: ";
      for (const auto pointedTo : element->pointsTo) {
        std::cout << pointedTo->value->uniqueName() << ", ";
      }
      std::cout << "\n";
    }
  }

  std::cout << "\n===3. WILDCARDS===\n";
  for (const auto wildcard : wildcards_) {
    std::cout << wildcard->uniqueName() << ", ";
  }
  std::cout << "\n";
}

// Do a breadth-first search over the graph, starting at `this` and
// traversing in the direction `dir`.`fn` will be run on each element.
//
// If `shortCircuit` is set, then if `fn` evaluates to true the search will
// short-circuit and return true. You can use this to do existence checks
// on the graph or whatever.
template <typename Fn>
bool AliasTracker::Element::bfs(Fn fn, BfsDirection dir, bool shortCircuit)
    const {
  std::queue<const Element*> queue;
  std::unordered_set<const Element*> seen;

  queue.push(this);
  while (!queue.empty()) {
    const auto el = queue.front();
    queue.pop();
    seen.insert(el);

    if (fn(el) && shortCircuit) {
      return true;
    }

    switch (dir) {
      case BfsDirection::POINTS_TO: {
        for (auto ptr : el->pointsTo) {
          if (!seen.count(ptr)) {
            queue.push(ptr);
          }
        }
      } break;

      case BfsDirection::POINTED_FROM: {
        for (auto ptr : el->pointedFrom) {
          if (!seen.count(ptr)) {
            queue.push(ptr);
          }
        }
      } break;

      case BfsDirection::BOTH: {
        for (auto ptr : el->pointsTo) {
          if (!seen.count(ptr)) {
            queue.push(ptr);
          }
        }
        for (auto ptr : el->pointedFrom) {
          if (!seen.count(ptr)) {
            queue.push(ptr);
          }
        }
      } break;
    }
  }
  return false;
}
// Cache results in a way to make common queries constant time.
void AliasTracker::cache() const {
  if (!cacheStale_) {
    return;
  }

  for (const auto& pr : elements_) {
    const auto el = pr.first;
    // For each value that does point to anything, assign a fresh set.
    if (el->pointsTo.size() == 0) {
      const auto id = getFreshId();
      assignSet(el, id);

      // Propagate this set to every element that points to `el`
      el->bfs(
          [&](const Element* pointerTo) { return assignSet(pointerTo, id); },
          BfsDirection::POINTED_FROM);
    }
  }

  cacheStale_ = false;
}

bool AliasTracker::hasWritersCached(const Value* v) const {
  cache();
  for (const auto& set : elementToSet_.at(map_.at(v))) {
    if (setToWrites_.count(set) && setToWrites_.at(set).size() > 0) {
      return true;
    }
  }
  return false;
}

std::unordered_set<Node*> AliasTracker::getWritersCached(const Value* v) const {
  cache();
  std::unordered_set<Node*> ret;
  for (const auto& set : elementToSet_.at(map_.at(v))) {
    if (setToWrites_.count(set) > 0) {
      for (auto write : setToWrites_.at(set)) {
        ret.insert(write);
      }
    }
  }
  return ret;
}

bool AliasTracker::assignSet(const Element* el, set_id_t id) const {
  elementToSet_[el].insert(id);
  for (auto write : el->writers) {
    setToWrites_[id].insert(write);
  }
  return true;
}

} // namespace jit
} // namespace torch
