#include "memory_dag.h"

#include <c10/util/flat_hash_map.h>
#include <torch/csrc/utils/memory.h>
#include <algorithm>
#include <queue>

namespace torch {
namespace jit {

Element::Element(MemoryDAG& dag_, const Value* value_, unsigned index_)
    : dag(dag_), index(index_), value(value_) {}

const Element* MemoryDAG::fromIndex(unsigned x) const {
  TORCH_INTERNAL_ASSERT(x < indexToElementMap_.size());
  return indexToElementMap_[x].get();
}

Element* MemoryDAG::fromIndex(unsigned x) {
  TORCH_INTERNAL_ASSERT(x < indexToElementMap_.size());
  return indexToElementMap_[x].get();
}

bool MemoryDAG::mayAlias(Element* a, Element* b) const {
  return mayAliasImpl(a, b);
}

bool MemoryDAG::mayAlias(const Element* a, const Element* b) const {
  return mayAliasImpl(a, b);
}

bool MemoryDAG::mayAliasImpl(const Element* a, const Element* b) const {
  const auto aMemLoc = a->getMemoryLocations();
  const auto bMemLoc = b->getMemoryLocations();

  return aMemLoc.intersects(bMemLoc);
}

bool MemoryDAG::mayContainAlias(const Element* a, const Element* b) const {
  return mayContainAliasImpl(a, b);
}

bool MemoryDAG::mayContainAlias(Element* a, Element* b) const {
  return mayContainAliasImpl(a, b);
}

void MemoryDAG::collectAllContainedMemoryLocations(
    const Element* elem,
    MemoryLocations& cont) const {
  // we have already recursed on this element
  unsigned compIdx = elem->index;
  if (cont.test(compIdx)) {
    return;
  }
  cont.set(compIdx);

  for (const auto& mem_loc : elem->getMemoryLocations()) {
    collectAllContainedMemoryLocations(fromIndex(mem_loc), cont);
  }

  for (const auto& contained : elem->containedElements) {
    collectAllContainedMemoryLocations(fromIndex(contained), cont);
  }
}

bool MemoryDAG::mayContainAliasImpl(const Element* a, const Element* b) const {
  MemoryLocations all_a_mlocs;
  MemoryLocations all_b_mlocs;

  collectAllContainedMemoryLocations(a, all_a_mlocs);
  collectAllContainedMemoryLocations(b, all_b_mlocs);

  return all_a_mlocs.intersects(all_b_mlocs);
}

bool MemoryDAG::mayContainAlias(
    const at::ArrayRef<Element*> a,
    const at::ArrayRef<Element*> b) const {
  if (a.size() == 0 || b.size() == 0) {
    return false;
  }

  MemoryLocations all_a_mlocs;
  for (const auto& elem : a) {
    collectAllContainedMemoryLocations(elem, all_a_mlocs);
  }

  MemoryLocations all_b_mlocs;
  for (const auto& elem : b) {
    collectAllContainedMemoryLocations(elem, all_b_mlocs);
  }

  return all_a_mlocs.intersects(all_b_mlocs);
}

void MemoryDAG::makePointerTo(Element* from, Element* to) {
  from->pointsTo.set(to->index);
  from->cachedMemoryLocations_.clear();

  to->pointedFrom.set(from->index);
}

void MemoryDAG::addToContainedElements(Element* elem, Element* container) {
  TORCH_INTERNAL_ASSERT(
      elem != container, "Elements cannot contain themselves");
  container->containedElements.set(elem->index);
}

// Give `v` a fresh alias (i.e. it does not point to any value)
Element* MemoryDAG::makeFreshValue(const Value* v) {
  indexToElementMap_.emplace_back(
    torch::make_unique<Element>(*this, v, indexToElementMap_.size()));
  return indexToElementMap_.back().get();
}

const MemoryLocations& Element::getMemoryLocations() const {
  if (!cachedMemoryLocations_.empty()) {
    return cachedMemoryLocations_;
  }

  // Do a BFS in the `points-to` direction, collecting all memory locations
  MemoryLocations ret;
  this->bfs(BfsDirection::POINTS_TO, ret);
  cachedMemoryLocations_ = ret;
  return cachedMemoryLocations_;
}

// Do a breadth-first search over the graph, starting at `this` and
// traversing in the direction `dir`.`fn` will be run on each element.
void Element::bfs(BfsDirection dir, MemoryLocations& res) const {
  std::queue<unsigned> queue;
  ska::flat_hash_set<int> seen;
  queue.push(this->index);
  while (!queue.empty()) {
    const auto index = queue.front();
    queue.pop();
    seen.insert(index);
    auto el = dag.fromIndex(index);
    if (el->pointsTo.empty()) {
      res.set(index);
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
    }
  }
}
} // namespace jit
} // namespace torch
