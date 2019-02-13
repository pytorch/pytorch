#include "alias_tracker.h"

#include <torch/csrc/utils/memory.h>
#include <queue>

namespace torch {
namespace jit {

// Returns true iff `v` is present in the alias set tracker.
bool AliasTracker::contains(const Value* v) const {
  return isWildcard(v) || map_.count(v);
}

bool AliasTracker::mayAlias(const Value* a, const Value* b) const {
  if (isWildcard(a) || isWildcard(b)) {
    return true;
  }

  if (!map_.count(a) || !map_.count(b)) {
    return false;
  }

  const auto aEl = map_.at(a);
  const auto bEl = map_.at(b);

  const auto aMemLoc = aEl->getMemoryLocations();
  const auto bMemLoc = bEl->getMemoryLocations();

  // XXX: This could be more efficiently done as a bitwise AND on two bitfields
  // that represent memory location membership. If these comparisons end up
  // being a bottleneck, consider implementing it that way.
  for (const auto aLoc : aMemLoc) {
    for (const auto bLoc : bMemLoc) {
      if (aLoc == bLoc) {
        return true;
      }
    }
  }
  return false;
}

bool AliasTracker::writesTo(Node* n, const Value* v) const {
  if (isWildcard(v)) {
    return wildcardWriters_.count(n);
  }

  if (!map_.count(v) || !writeIndex_.count(n)) {
    return false;
  }

  // Can short-circuit if we know this node writes directly to `v`
  if (writeIndex_.at(n).count(v)) {
    return true;
  }

  // Otherwise, check if `v` may alias any of written-to values in `n`
  const auto vSet = ValueSet{v};
  return mayAlias(vSet, writeIndex_.at(n));
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
  writeIndex_[n].insert(v);
}

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

  if (isWriteCacheStale_) {
    rebuildWriteCache();
  }

  for (const auto loc : map_.at(v)->getMemoryLocations()) {
    if (writeCache_.count(loc)) {
      return true;
    }
  }

  return false;
}

void AliasTracker::rebuildWriteCache() const {
  for (const auto& pr : writeIndex_) {
    const auto& writtenValues = pr.second;

    for (const auto value : writtenValues) {
      for (const auto loc : map_.at(value)->getMemoryLocations()) {
        writeCache_.insert(loc);
      }
    }
  }
  isWriteCacheStale_ = false;
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

std::unordered_set<const AliasTracker::Element*> AliasTracker::Element::
    getMemoryLocations() const {
  if (!cachedMemoryLocations_.empty()) {
    return cachedMemoryLocations_;
  }

  // Do a BFS in the `points-to` direction, collecting all memory locations
  std::unordered_set<const Element*> ret;
  this->bfs(
      [&](const Element* el) {
        if (el->pointsTo.empty()) {
          ret.insert(el);
        }
      },
      BfsDirection::POINTS_TO);

  cachedMemoryLocations_ = ret;
  return ret;
}

// Do a breadth-first search over the graph, starting at `this` and
// traversing in the direction `dir`.`fn` will be run on each element.
template <typename Fn>
bool AliasTracker::Element::bfs(Fn fn, BfsDirection dir) const {
  std::queue<const Element*> queue;
  std::unordered_set<const Element*> seen;

  queue.push(this);
  while (!queue.empty()) {
    const auto el = queue.front();
    queue.pop();
    seen.insert(el);

    fn(el);

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
    }
  }
  return false;
}

ValueSet AliasTracker::getMemoryLocations(const Value* v) const {
  ValueSet ret;
  if (!map_.count(v)) {
    return ret;
  }

  for (const auto el : map_.at(v)->getMemoryLocations()) {
    ret.insert(el->value);
  }
  return ret;
}
} // namespace jit
} // namespace torch
