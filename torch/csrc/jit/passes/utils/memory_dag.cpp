#include "memory_dag.h"

#include <torch/csrc/utils/memory.h>
#include <algorithm>
#include <iostream>
#include <queue>

namespace torch {
namespace jit {

bool MemoryDAG::mayAlias(Element* a, Element* b) const {
  return mayAliasImpl(a, b);
}

bool MemoryDAG::mayAlias(const Element* a, const Element* b) const {
  return mayAliasImpl(a, b);
}

bool MemoryDAG::mayAliasImpl(const Element* a, const Element* b) const {
  const auto aMemLoc = a->getMemoryLocations();
  const auto bMemLoc = b->getMemoryLocations();

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

bool MemoryDAG::mayContainAlias(const Element* a, const Element* b) const {
  return mayContainAliasImpl(a, b);
}

bool MemoryDAG::mayContainAlias(Element* a, Element* b) const {
  return mayContainAliasImpl(a, b);
}

void collectAllContainedMemoryLocations(
    const Element* elem,
    std::unordered_set<const Element*>& cont) {
  // we have already recursed on this element
  if (cont.count(elem)) {
    return;
  }

  cont.insert(elem);

  for (const auto& mem_loc : elem->getMemoryLocations()) {
    collectAllContainedMemoryLocations(mem_loc, cont);
  }

  for (const auto& contained : elem->contained_elements) {
    collectAllContainedMemoryLocations(contained, cont);
  }
}

bool MemoryDAG::mayContainAliasImpl(const Element* a, const Element* b) const {
  std::unordered_set<const Element*> all_a_mlocs;
  std::unordered_set<const Element*> all_b_mlocs;

  collectAllContainedMemoryLocations(a, all_a_mlocs);
  collectAllContainedMemoryLocations(b, all_b_mlocs);

  for (const auto a_mem : all_a_mlocs) {
    for (const auto b_mem : all_b_mlocs) {
      if (a_mem == b_mem) {
        return true;
      }
    }
  }

  return false;
}

// Make `v` point at `to`.
void MemoryDAG::makePointerTo(Element* from, Element* to) {
  from->pointsTo.insert(to);
  to->pointedFrom.insert(from);
}

void MemoryDAG::addToContainedElements(Element* elem, Element* container) {
  container->contained_elements.insert(elem);
}

// Give `v` a fresh alias (i.e. it does not point to any value)
Element* MemoryDAG::makeFreshValue(const Value* v) {
  auto el = torch::make_unique<Element>();
  el->value = v;

  auto rawPtr = el.get();
  elements_.emplace(rawPtr, std::move(el));
  return rawPtr;
}

std::unordered_set<const Element*> Element::getMemoryLocations() const {
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
bool Element::bfs(Fn fn, BfsDirection dir) const {
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
} // namespace jit
} // namespace torch
